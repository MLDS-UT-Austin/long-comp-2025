from abc import ABC, abstractmethod
from typing import Callable, final

from data import *
from llm import DummyLLM, LLMProxy, LLMTokenCounterWrapper


class Agent(ABC):
    """This is the base class for all agents. You should subclass this class and implement the abstract methods.
    Note: Locations are represented by the Location enum
    Note: Players are indexed from 0 to n_players - 1, inclusive with player 0 being you
    """

    @abstractmethod
    def __init__(
        self,
        location: Location | None,
        n_players: int,
        n_rounds: int,
        llm: LLMProxy,
    ) -> None:
        """Args:
        location (Location | None): the enum location for the game or None if the agent is the spy
        n_players (int): number of players including yourself
        n_rounds (int): total number rounds. Each round includes a question, answer, and a vote.
        """
        pass

    @abstractmethod
    async def ask_question(self) -> tuple[int, str]:
        """This method is called when it is the agent's turn to ask a question.

        Returns:
            tuple[int, str]: the index of the opponent to ask the question and the question to ask
                Note: opponent indices range from 1 to n_players - 1, inclusive
        """
        return 1, "question here"

    @abstractmethod
    async def answer_question(self, question: str) -> str:
        """This method is called when the agent is asked a question.

        Args:
            question (str): the question asked by the opponent

        Returns:
            str: your answer to the question
        """
        return "answer here"

    @abstractmethod
    async def analyze_response(
        self,
        questioner: int,
        question: str,
        answerer: int,
        response: str,
    ) -> None:
        """This method is called every round after a question is asked and answered.

        Args:
            questioner (int): the index of the opponent who asked the question
            question (str): the question asked by the opponent
            answerer (int): the index of the opponent who answered the question
            response (str): the response to the question
        """
        pass

    @abstractmethod
    async def guess_location(self) -> Location | None:
        """This method is called every round when the agent is the spy

        Returns:
            Location | None: The location guessed by the agent or None if the agent does not want to guess
                Note: The agent can only guess once per game
        """
        return Location.AIRPLANE

    @abstractmethod
    async def accuse_player(self) -> int | None:
        """This method is called at the end of every round

        Returns:
            int | None: The index of the opponent the agent accuses of being the spy or None if the agent does not want to accuse
        """
        return 1

    @abstractmethod
    async def analyze_voting(self, votes: list[int | None]) -> None:
        """This method is called at the end of every round after all players have voted
        Args:
            votes (list[int  |  None]): a list containing the vote for each player
                Ex: [0, None, ...] means player 0 voted for you, player 1 did not vote, etc.
        """
        pass

    @staticmethod
    @final
    def validate(agent: "Agent") -> None:
        """Checks agent inputs and outputs. Run this method to check if your agent is working correctly."""
        # TODO
        pass


class ExampleAgent(Agent):
    def __init__(
        self, location: Location | None, n_players: int, n_rounds: int, llm: LLMProxy
    ) -> None:
        self.location = location
        self.n_players = n_players
        self.n_rounds = n_rounds
        self.llm = llm

    async def ask_question(self) -> tuple[int, str]:
        # use await in front of self.llm.prompt() because it is an async function
        question = await self.llm.prompt(
            f"I am playing spyfall at the {self.location}. Do not reveal the location. What is a good question to ask?"
        )
        return 1, question


if __name__ == "__main__":
    agent = ExampleAgent(Location.AIRPLANE, 4, 20, LLMProxy())
    Agent.validate(agent)
