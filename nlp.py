import asyncio
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch
from dotenv import load_dotenv
from together import AsyncTogether  # type: ignore
from transformers import (  # type: ignore
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# TODO: get rid of "llm" in other files
# Please set your API key in the .env file: "TOGETHER_API_KEY=<your-api-key>"
load_dotenv()

try:
    client = AsyncTogether()
except Exception as e:
    client = e


class LLMRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


# Interfaces ####################################################################


class LLMTokenizer(ABC):
    @abstractmethod
    def count_tokens(self, text_or_prompt: str | list[tuple[LLMRole, str]]) -> int:
        pass


class LLM(ABC):
    @abstractmethod
    async def prompt(
        self, prompt: list[tuple[LLMRole, str]], max_output_tokens: int | None = None
    ) -> str:
        pass


class EmbeddingTokenizer(ABC):
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass


class Embedding(ABC):
    @abstractmethod
    async def get_embeddings(self, text: str) -> np.ndarray:
        pass


# Implementations ####################################################################


class LlamaTokenizer(LLMTokenizer):
    def __init__(self):
        # Use NousResearch bc it doesn't have retricted access
        self.tokenizer = AutoTokenizer.from_pretrained(
            "NousResearch/Meta-Llama-3-8B-Instruct"
        )

    def count_tokens(self, text_or_prompt: str | list[tuple[LLMRole, str]]) -> int:
        if isinstance(text_or_prompt, str):
            tokens = self.tokenizer(text_or_prompt).encodings[0].tokens
            return len(tokens)
        else:
            tokens = [
                self.tokenizer(content).encodings[0].tokens
                for role, content in text_or_prompt
            ]
            return sum(len(token) for token in tokens)


class Llama(LLM):
    def __init__(self):
        super().__init__()
        if isinstance(client, Exception):
            raise client

    async def prompt(
        self, prompt: list[tuple[LLMRole, str]], max_output_tokens: int | None = None
    ) -> str:
        messages = [
            {"role": role.value, "content": content} for role, content in prompt
        ]
        response = await client.chat.completions.create(
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


class DummyLLM(LLM):
    """A dummy LLM for testing"""

    async def prompt(
        self, prompt: list[tuple[LLMRole, str]], max_output_tokens: int | None = None
    ) -> str:
        return "Output from the LLM will be here"


class BERTTokenizer(EmbeddingTokenizer):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased", clean_up_tokenization_spaces=True
        )

    def count_tokens(self, text: str) -> int:
        tokens = self.tokenizer(text).encodings[0].tokens
        return len(tokens)


class BERTTogether(Embedding):
    def __init__(self):
        super().__init__()
        if isinstance(client, Exception):
            raise client

    async def get_embeddings(self, text: str) -> np.ndarray:
        """returns a 768-dimensional embedding"""
        response = await client.embeddings.create(
            model="togethercomputer/m2-bert-80M-2k-retrieval",
            input=text,
        )
        embedding = np.array(response.data[0].embedding)
        return embedding


class BERTLocal(Embedding):
    def __init__(
        self,
        device: str = "cpu",
        batch_size: int = 8,
    ):
        self.device = device
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            clean_up_tokenization_spaces=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "togethercomputer/m2-bert-80M-2k-retrieval", trust_remote_code=True
        ).to(device)
        self.model.eval()

    async def model_loop(self):
        """Use a loop to optimize with batching"""
        while True:
            dequeued = [await self.queue.get()]
            for _ in range(self.batch_size - 1):
                try:
                    dequeued.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            # print("batch size", len(dequeued))
            inputs, futures = zip(*dequeued)

            # tokenize and pad
            input_ids = self.tokenizer(
                inputs,
                return_tensors="pt",
                padding="longest",
                return_token_type_ids=False,
            ).to(self.device)

            # run model
            with torch.no_grad():
                outputs = self.model(**input_ids)

            embeddings = outputs["sentence_embedding"].detach().cpu().numpy()
            for future, embedding in zip(futures, embeddings):
                future.set_result(embedding)

    async def get_embeddings(self, text: str) -> np.ndarray:
        """This adds the text to the queue which will be processed in the model_loop
        the model_loop will then set the future with the embedding"""
        if not hasattr(self, "model_loop_task") or self.model_loop_task.cancelled():  # type: ignore
            self.queue: asyncio.Queue[tuple[str, asyncio.Future]] = asyncio.Queue()
            self.model_loop_task = asyncio.create_task(self.model_loop())
        future = asyncio.get_event_loop().create_future()
        self.queue.put_nowait((text, future))
        embedding = await future
        return embedding


class DummyEmbedding(Embedding):
    async def get_embeddings(self, text: str) -> np.ndarray:
        return np.zeros(768)


@dataclass
class NLP:
    """Used for the single, main NLP instance in the runtime"""

    llm_tokenizer: LLMTokenizer = field(default_factory=LlamaTokenizer)
    llm: LLM = field(default_factory=DummyLLM)
    embedding_tokenizer: EmbeddingTokenizer = field(default_factory=BERTTokenizer)
    embedding: Embedding = field(default_factory=DummyEmbedding)

    async def prompt_llm(
        self, prompt: list[tuple[LLMRole, str]], max_output_tokens: int | None = None
    ) -> str:
        return await self.llm.prompt(prompt, max_output_tokens)

    async def get_embeddings(self, text: str) -> np.ndarray:
        return await self.embedding.get_embeddings(text)

    def count_llm_tokens(self, text_or_prompt: str | list[tuple[LLMRole, str]]) -> int:
        return self.llm_tokenizer.count_tokens(text_or_prompt)

    def count_embedding_tokens(self, text: str) -> int:
        return self.embedding_tokenizer.count_tokens(text)


@dataclass
class TokenCounterWrapper:
    """Used for NLP instances that need to keep track of token usage"""

    nlp: NLP = field(default_factory=NLP)
    token_limit: int = 4096

    def __post_init__(self):
        self.reset_token_counter()

    def reset_token_counter(self):
        self.remaining_tokens = self.token_limit

    async def prompt_llm(
        self, prompt: list[tuple[LLMRole, str]], max_output_tokens: int | None = None
    ) -> str:
        if self.remaining_tokens is not None:
            self.remaining_tokens -= self.nlp.count_llm_tokens(prompt)
            if self.remaining_tokens <= 0:
                return ""

            if max_output_tokens is None:
                max_output_tokens = self.remaining_tokens
            else:
                max_output_tokens = min(max_output_tokens, self.remaining_tokens)

            output = await self.nlp.prompt_llm(prompt, max_output_tokens)
            self.remaining_tokens -= self.nlp.count_llm_tokens(output)
        return output

    async def get_embeddings(self, text: str) -> np.ndarray:
        self.remaining_tokens -= math.ceil(self.nlp.count_embedding_tokens(text) / 10)
        if self.remaining_tokens < 0:
            return np.zeros(768)

        return await self.nlp.get_embeddings(text)

    def count_llm_tokens(self, text_or_prompt: str | list[tuple[LLMRole, str]]) -> int:
        return self.nlp.count_llm_tokens(text_or_prompt)

    def count_embedding_tokens(self, text: str) -> int:
        return self.nlp.count_embedding_tokens(text)


class NLPProxy:
    """The wrapper that agents will use to interact with the LLM"""

    def __init__(self, token_counter: TokenCounterWrapper | None = None):
        if token_counter is None:
            token_counter = TokenCounterWrapper()
        self.__token_counter = token_counter

    async def prompt_llm(
        self, prompt: list[tuple[LLMRole, str]], max_output_tokens: int | None = None
    ) -> str:
        return await self.__token_counter.prompt_llm(prompt, max_output_tokens)

    async def get_embeddings(self, text: str) -> np.ndarray:
        # TODO docs, also explain variable costs
        return await self.__token_counter.get_embeddings(text)

    def count_llm_tokens(self, text_or_prompt: str | list[tuple[LLMRole, str]]) -> int:
        return self.__token_counter.count_llm_tokens(text_or_prompt)

    def count_embedding_tokens(self, text: str) -> int:
        return self.__token_counter.count_embedding_tokens(text)

    def get_remaining_tokens(self):
        return self.__token_counter.remaining_tokens


async def main():
    bert = BERTLocal()
    futures = [bert.get_embeddings("How many US states are there?") for _ in range(10)]
    await asyncio.gather(*futures)


if __name__ == "__main__":
    # llm = Llama()
    # prompt = [(LLMRole.USER, "How many US states are there?")]
    # output = asyncio.run(llm.prompt(prompt, 100))
    # print(output)

    # bert = BERTTogether()
    # output = asyncio.run(bert.get_embeddings("How many US states are there?"))
    # print(type(output))
    # print(output)

    # bert = BERTLocal()
    # output = asyncio.run(bert.get_embeddings("How many US states are there?"))
    # print(type(output))
    # output = asyncio.run(bert.get_embeddings("How many US states are there?"))
    # print(type(output))
    # output = asyncio.run(bert.get_embeddings("How many US states are there?"))
    # print(type(output))

    asyncio.run(main())
