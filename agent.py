import asyncio
from abc import ABC, abstractmethod
from typing import final

from data import Location
from nlp import NLPProxy

AGENT_REGISTRY = {}


def register_agent(cls):
    """Type @register_agent on top of the class definition to register the agent"""
    AGENT_REGISTRY[cls.__name__] = cls
    return cls


class Agent(ABC):
    """This is the base class for all agents
    You should subclass this class and implement the abstract methods in submission.py
    Note: Throughout this class, players are indexed from 0 to n_players - 1, inclusive with player 0 being you
          also, locations are represented by the Location enum
    """

    @abstractmethod
    def __init__(
        self,
        location: Location | None,
        n_players: int,
        n_rounds: int,
        nlp: NLPProxy,
    ) -> None:
        """Args:
        location (Location | None): the enum location for the game or None if the agent is the spy
        n_players (int): number of players including yourself
        n_rounds (int): total number rounds. Each round includes a question, answer, and a vote.
        nlp (NLPProxy): allows you to prompt the llm and get embeddings. You are given # TODO tokens per round.
            If you exceed the token limit, get_prompt will return an empty string and get_embeddings will return a 0 array.
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

    @classmethod
    @final
    def validate(cls) -> None:
        """Quick check for return types and edge cases"""
        agent = cls(Location.AIRPLANE, 5, 5, NLPProxy())
        answerer, question = asyncio.run(agent.ask_question())
        answer0 = asyncio.run(agent.answer_question("question here"))
        answer1 = asyncio.run(agent.answer_question(""))
        asyncio.run(agent.analyze_response(1, "question here", 2, "answer here"))
        asyncio.run(agent.analyze_response(0, "question here", 3, "answer here"))
        asyncio.run(agent.analyze_response(2, "", 0, ""))
        guess = asyncio.run(agent.guess_location())
        accusation = asyncio.run(agent.accuse_player())
        asyncio.run(agent.analyze_voting([0, 1, None, None]))
        assert isinstance(answerer, int)
        assert 1 <= answerer < 5
        assert isinstance(question, str)
        assert isinstance(answer0, str)
        assert isinstance(answer1, str)
        assert isinstance(guess, Location) or guess is None
        assert isinstance(accusation, int) or accusation is None
        if accusation is not None:
            assert 1 <= accusation < 5


# TODO: rewrite this as one of our agents
@register_agent
class ExampleAgent(Agent):
    def __init__(
        self,
        location: Location | None,
        n_players: int,
        n_rounds: int,
        nlp: NLPProxy,
    ) -> None:
        self.location = location
        self.n_players = n_players
        self.n_rounds = n_rounds
        self.nlp = nlp

    async def ask_question(self) -> tuple[int, str]:
        return 1, "question"

    async def answer_question(self, question: str) -> str:
        return "answer"

    async def analyze_response(
        self,
        questioner: int,
        question: str,
        answerer: int,
        response: str,
    ) -> None:
        pass

    async def guess_location(self) -> Location | None:
        return None

    async def accuse_player(self) -> int | None:
        return None

    async def analyze_voting(self, votes: list[int | None]) -> None:
        pass
