import os

import numpy as np
from dotenv import load_dotenv
from transformers import AutoTokenizer

model_name = 'facebook/llama-7b'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Please set your OpenAI API key in the .env file: "API_KEY=your-api-key"
load_dotenv()
API_KEY = os.getenv("API_KEY")


class LLM:
    def __init__(self):
        # Use NousResearch bc it doesn't have retricted access
        self.tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct")

    async def prompt(self, input: str, max_output_tokens: int | None = None) -> str:
        raise NotImplementedError()  # TODO

    async def get_embeddings(self, input: str) -> np.ndarray:
        raise NotImplementedError()  # TODO

    def count_tokens(self, input: str) -> int:
        tokens = tokenizer(input).encodings[0].tokens
        # print(tokens)
        return len(tokens)


class DummyLLM(LLM):
    """A dummy LLM for testing"""

    async def prompt(self, input: str, max_output_tokens: int | None = None) -> str:
        return "text that the llm returns"

    async def get_embeddings(self, input: str) -> np.ndarray:
        return np.zeros(512)


class CopyCatLLM(LLM):
    """A LLM for estimating token usage"""

    async def prompt(self, input: str, max_output_tokens: int | None = None) -> str:
        return input

    async def get_embeddings(self, input: str) -> np.ndarray:
        return np.zeros(512)


class TokenCounterWrapper(LLM):
    """A wrapper around an LLM that keeps track of the number of tokens used per round"""

    def __init__(self, llm: LLM, token_limit: int = 4096):
        """_summary_

        Args:
            llm (LLM): the llm that is wrapped
            token_limit (int): the maximum number of tokens that can be used per agent per round
        """
        self.llm = llm
        self.token_limit = token_limit
        self.remaining_tokens = token_limit

    def reset_token_counter(self):
        self.remaining_tokens = self.token_limit

    async def prompt(self, input: str, max_output_tokens: int | None = None) -> str:
        self.remaining_tokens -= self.count_tokens(input)
        if self.remaining_tokens < 0:
            return ""

        if max_output_tokens is None:
            max_output_tokens = self.remaining_tokens
        else:
            max_output_tokens = min(max_output_tokens, self.remaining_tokens)

        return await self.llm.prompt(input, max_output_tokens)

    async def get_embeddings(self, input: str) -> np.ndarray:
        self.remaining_tokens -= self.llm.count_tokens(input)
        if self.remaining_tokens < 0:
            return np.zeros(0)

        return await self.llm.get_embeddings(input)

    def count_tokens(self, input: str) -> int:
        return self.llm.count_tokens(input)


class LLMProxy:
    """The wrapper that agents will use to interact with the LLM"""

    def __init__(self, llm: TokenCounterWrapper | None = None):
        if llm is None:
            llm = TokenCounterWrapper(DummyLLM())
        self.__llm = llm

    async def prompt(self, input: str, max_output_tokens: int | None = None) -> str:
        return await self.__llm.prompt(input, max_output_tokens)

    async def get_embeddings(self, input: str) -> np.ndarray:
        return await self.__llm.get_embeddings(input)

    def count_tokens(self, input: str) -> int:
        return self.__llm.count_tokens(input)

    def get_remaining_tokens(self):
        return self.__llm.remaining_tokens
