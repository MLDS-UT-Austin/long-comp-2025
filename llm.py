import numpy as np


class LLM:
    async def prompt_llm(self, input: str, max_output_tokens: int | None = None) -> str:
        return "llm output"

    async def get_embeddings(self, input: str) -> np.ndarray:
        return np.zeros(64)
