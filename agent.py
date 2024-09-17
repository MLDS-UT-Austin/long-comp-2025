from abc import ABC, abstractmethod
from enum import Enum

from util import *


class Location(Enum):
    PARK = "park"
    MALL = "mall"
    OFFICE = "office"
    SCHOOL = "school"


# template for the Agent class
class Agent(ABC):
    def __init__(self, spy: bool, n_players: int, location: Location):
        pass

    @abstractmethod
    def ask_question(self) -> tuple[int, str]:
        return 0, "question"

    @abstractmethod
    def answer_question(self, question: str) -> str:
        return "answer"

    @abstractmethod
    def analyze_response(
        self,
        questioner: int,
        question: str,
        answerer: int,
        response: str,
    ) -> None:
        spy = prompt_llm(
            f" are they the spy if they answered {question} this way? {response}"
        )

    @abstractmethod
    def guess_location(self) -> Location | None:
        return None

    @abstractmethod
    def accuse_player(self) -> int | None:
        return True

    @abstractmethod
    def analyze_voting(self, votes: list[int | None]) -> None:
        pass

    def validate(self):
        pass


class ExampleAgent(Agent):
    pass
