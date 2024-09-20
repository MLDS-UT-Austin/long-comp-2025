import asyncio
import math
import os
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from dotenv import load_dotenv
from together import AsyncTogether
from transformers import AutoTokenizer  # type: ignore

# Please set your API key in the .env file: "API_KEY=<your-api-key>"
load_dotenv()


class LLMRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class LLM(ABC):
    def __init__(self):
        # Use NousResearch bc it doesn't have retricted access
        self.tokenizer = AutoTokenizer.from_pretrained(
            "NousResearch/Meta-Llama-3-8B-Instruct"
        )

    @abstractmethod
    async def prompt(
        self, prompt: list[tuple[LLMRole, str]], max_output_tokens: int | None = None
    ) -> str:
        pass

    @abstractmethod  # TODO: make embeddings 10x cheaper?
    async def get_embeddings(self, text: str) -> np.ndarray:
        pass

    def count_tokens(self, text_or_prompt: str | list[tuple[LLMRole, str]]) -> int:
        if isinstance(text_or_prompt, str):
            tokens = self.tokenizer(text_or_prompt).encodings[0].tokens
            # print(tokens)
            return len(tokens)
        else:
            tokens = [
                self.tokenizer(content).encodings[0].tokens
                for role, content in text_or_prompt
            ]
            # print(tokens)
            return sum(len(token) for token in tokens)


class TogetherLLM(LLM):
    def __init__(self):
        super().__init__()
        self.client = AsyncTogether(api_key=os.getenv("API_KEY"))

    async def prompt(
        self, prompt: list[tuple[LLMRole, str]], max_output_tokens: int | None = None
    ) -> str:
        messages = [
            {"role": role.value, "content": content} for role, content in prompt
        ]
        response = await self.client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct-Lite",
            messages=messages,
            max_tokens=max_output_tokens,
            temperature=0.7,  # TODO: make adjustable?
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|eot_id|>"],
            stream=False,
        )
        return response.choices[0].message.content

    async def get_embeddings(self, text: str) -> np.ndarray:
        return np.zeros(512)


class DummyLLM(LLM):
    """A dummy LLM for testing"""

    async def prompt(
        self, prompt: list[tuple[LLMRole, str]], max_output_tokens: int | None = None
    ) -> str:
        return "Output from the LLM will be here"

    async def get_embeddings(self, text: str) -> np.ndarray:
        return np.zeros(512)


class CopyCatLLM(LLM):
    """A LLM for estimating token usage"""

    async def prompt(
        self, prompt: list[tuple[LLMRole, str]], max_output_tokens: int | None = None
    ) -> str:
        return prompt[-1][1]

    async def get_embeddings(self, text: str) -> np.ndarray:
        return np.zeros(512)


class TokenCounterWrapper(LLM):
    """A wrapper around an LLM that keeps track of the number of tokens used per round"""

    def __init__(self, llm: LLM, token_limit: int = 4096): # FIXME: 4096?
        super().__init__()
        self.llm = llm
        self.token_limit = token_limit
        self.remaining_tokens = token_limit

    def reset_token_counter(self):
        self.remaining_tokens = self.token_limit

    async def prompt(
        self, prompt: list[tuple[LLMRole, str]], max_output_tokens: int | None = None
    ) -> str:
        self.remaining_tokens -= self.count_tokens(prompt)
        if self.remaining_tokens <= 0:
            return ""

        if max_output_tokens is None:
            max_output_tokens = self.remaining_tokens
        else:
            max_output_tokens = min(max_output_tokens, self.remaining_tokens)

        output = await self.llm.prompt(prompt, max_output_tokens)
        self.remaining_tokens -= self.count_tokens(output)
        return output

    async def get_embeddings(self, text: str) -> np.ndarray:
        self.remaining_tokens -= math.ceil(self.llm.count_tokens(text) / 10)
        if self.remaining_tokens < 0:
            return np.zeros(0)

        return await self.llm.get_embeddings(text)


class LLMProxy:
    """The wrapper that agents will use to interact with the LLM"""

    def __init__(self, llm: TokenCounterWrapper | None = None):
        if llm is None:
            llm = TokenCounterWrapper(DummyLLM())
        self.__llm = llm

    async def prompt(
        self, prompt: list[tuple[LLMRole, str]], max_output_tokens: int | None = None
    ) -> str:
        # TODO docs, also explain variable costs
        return await self.__llm.prompt(prompt, max_output_tokens)

    async def get_embeddings(self, text: str) -> np.ndarray:
        return await self.__llm.get_embeddings(text)

    def count_tokens(self, text_or_prompt: str | list[tuple[LLMRole, str]]) -> int:

        return self.__llm.count_tokens(text_or_prompt)

    def get_remaining_tokens(self):
        return self.__llm.remaining_tokens


if __name__ == "__main__":
    llm = TokenCounterWrapper(CopyCatLLM())
    prompt = [(LLMRole.USER, "How many US states are there?")]
    print(llm.count_tokens(prompt))
    # output = asyncio.run(llm.prompt(prompt, 100))
    print(output)
    print(llm.count_tokens(output))
    print(llm.remaining_tokens)
