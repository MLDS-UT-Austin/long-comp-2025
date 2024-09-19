import asyncio
import random
from collections import Counter
from dataclasses import dataclass
from enum import Enum

from tqdm import tqdm

from agent import Agent
from data import *
from llm import *
from util import *


class GameState(Enum):
    RUNNING = "ongoing"
    SPY_GUESSED_RIGHT = "spy guessed right"
    SPY_GUESSED_WRONG = "spy guessed wrong"
    SPY_INDICTED = "spy indicted"
    NON_SPY_INDICTED = "non-spy indicted"
    NO_ONE_INDICTED = "no one indicted"


class Game:
    location: Location
    spy: int
    questioner: int
    player_classes: list[type[Agent]]
    players: list[Agent]
    player_llms: list[TokenCounterWrapper]
    game_state: GameState
    tqdm_bar: tqdm | None = None
    n_players: int

    spy_scoring = {
        GameState.RUNNING: 0,
        GameState.SPY_GUESSED_RIGHT: 4,
        GameState.SPY_GUESSED_WRONG: 0,
        GameState.SPY_INDICTED: 0,
        GameState.NON_SPY_INDICTED: 4,
        GameState.NO_ONE_INDICTED: 2,
    }
    nonspy_scoring = {
        GameState.RUNNING: 0,
        GameState.SPY_GUESSED_RIGHT: 0,
        GameState.SPY_GUESSED_WRONG: 1,
        GameState.SPY_INDICTED: 1,
        GameState.NON_SPY_INDICTED: 0,
        GameState.NO_ONE_INDICTED: 0,
    }

    def __init__(
        self,
        player_classes: list[type[Agent]],
        llm: LLM,
        n_rounds: int = 20,
    ):
        self.player_classes = player_classes
        n_players = self.n_players = len(player_classes)
        self.n_rounds = n_rounds

        self.location = random.choice(list(Location))
        self.spy = random.randint(0, n_players - 1)
        self.questioner = random.randint(0, n_players - 1)
        self.players = []
        self.player_llms = []
        self.game_state = GameState.RUNNING

        for i, player_class in enumerate(player_classes):
            player_llm = TokenCounterWrapper(llm)
            given_location = self.location if i != self.spy else None
            player = player_class(
                given_location, n_players, n_rounds, llm=LLMProxy(player_llm)
            )
            self.players.append(player)
            self.player_llms.append(player_llm)

        self.povs = [list(range(1, n_players)) for _ in range(n_players)]
        for i, pov in enumerate(self.povs):
            random.shuffle(pov)
            pov.insert(i, 0)
        self.r_povs = [[0] * (n_players) for _ in range(n_players)]
        for i in range(n_players):
            for player, player_w_pov in enumerate(self.povs[i]):
                self.r_povs[i][player_w_pov] = player

        self.rounds: list[Round] = []
        for i in range(n_players):  # TODO move to test cases
            for j in range(n_players):
                assert self.add_pov(self.reverse_pov(j, pov=i), pov=i) == j
                assert self.reverse_pov(self.add_pov(j, pov=i), pov=i) == j

    def add_pov(self, player: int, pov: int):
        return self.povs[pov][player]

    def reverse_pov(self, player: int, pov: int):
        return self.r_povs[pov][player]

    async def play(self):
        for _ in range(self.n_rounds):
            round = Round(self)
            await round.play()
            if self.tqdm_bar is not None:
                self.tqdm_bar.update(1)
            self.rounds.append(round)
            if self.game_state != GameState.RUNNING:
                return
        if self.tqdm_bar is not None:
            self.tqdm_bar.update(self.n_rounds - len(self.rounds))
        self.game_state = GameState.NO_ONE_INDICTED

    def get_scores(self) -> list[int]:
        scores = [self.nonspy_scoring[self.game_state]] * self.n_players
        scores[self.spy] = self.spy_scoring[self.game_state]
        return scores

    def render(self):
        # init pygame
        self.window = None
        for round in self.rounds:
            round.render()
        # close pygame


class Round:
    questioner: int
    question: str
    answerer: int
    answer: str

    spy_guess: Location | None

    player_votes: list[int | None]
    indicted: int | None

    def __init__(self, game: Game):
        self.game = game

    async def play(self):
        game = self.game
        for llm in game.player_llms:
            llm.reset_token_counter()
        questioner = self.questioner = game.questioner

        # TODO: should we increase the token count for the questioner and answerer?
        answerer, question = await game.players[questioner].ask_question()
        assert 1 <= answerer < game.n_players and isinstance(question, str)
        answerer = game.reverse_pov(answerer, pov=questioner)
        answer = await game.players[answerer].answer_question(question)
        assert isinstance(answer, str)
        futures = []
        for player in range(game.n_players):
            q = game.add_pov(questioner, pov=player)
            a = game.add_pov(answerer, pov=player)
            futures.append(
                game.players[player].analyze_response(q, question, a, answer)
            )
        await asyncio.gather(*futures)

        self.question = question
        self.answer = answer
        game.questioner = self.answerer = answerer

        # spy voting
        guess = self.spy_guess = await game.players[game.spy].guess_location()
        assert guess is None or isinstance(guess, Location)
        if guess == game.location:
            game.game_state = GameState.SPY_GUESSED_RIGHT
            return
        elif guess != None:
            game.game_state = GameState.SPY_GUESSED_WRONG
            return

        # player voting
        votes = self.player_votes = await asyncio.gather(
            *[player.accuse_player() for player in game.players]
        )
        assert all(1 <= vote < game.n_players for vote in votes if vote is not None)

        for i, vote in enumerate(votes):
            if vote is not None:
                votes[i] = game.reverse_pov(vote, pov=i)

        indicted = self.indicted = count_votes(votes, game.n_players)
        if indicted == game.spy:
            game.game_state = GameState.SPY_INDICTED
        elif indicted is not None:
            game.game_state = GameState.NON_SPY_INDICTED
        futures = []
        for i in range(game.n_players):
            votes_pov = [] * game.n_players
            for voter, votee in enumerate(votes_pov):
                voter = game.add_pov(voter, pov=i)
                votee = game.add_pov(votee, pov=i)
                votes_pov[voter] = votee
            futures.append(game.players[i].analyze_voting(votes_pov))
        await asyncio.gather(*futures)

    def get_conversation(self) -> list[tuple[int, str]]:
        """returns the conversation as a list of tuples of player index and their message"""
        game = self.game

        output = []
        output.append((self.questioner, f"Player {self.answerer + 1}, {self.question}"))
        output.append((self.answerer, self.answer))
        if self.spy_guess is not None:
            # spy: I am the spy. Was it the {location}?
            msg = random.choice(SPY_REVEAL_AND_GUESS).format(
                location=self.spy_guess.value
            )
            output.append((game.spy, msg))
            responder = random.choice(list(set(range(game.n_players)) - {game.spy}))
            if game.game_state == GameState.SPY_GUESSED_RIGHT:
                # random nonspy: yes that is right
                msg = random.choice(SPY_GUESS_RIGHT_RESPONSE)
            else:
                msg = random.choice(SPY_GUESS_WRONG_RESPONSE).format(
                    location=game.location.value
                )

            output.append((responder, msg))

        # write logic for indictions here if there is a majority
        if self.indicted is not None:
            # one of the accusers: "I think it's player {spy} are you the spy?"
            accuser = random.choice(
                [i for i, x in enumerate(self.player_votes) if x == game.spy]
            )
            msg = random.choice(ACCUSATION).format(spy=game.spy + 1)
            output.append((accuser, msg))
            if game.game_state == GameState.SPY_INDICTED:
                # spy: I am the spy
                msg = random.choice(SPY_INDICTED_RESPONSE)
                output.append((game.spy, msg))
            else:
                # indicted: No, I am not the spy
                msg = random.choice(NON_SPY_INDICTED_RESPONSE)
                output.append((self.indicted, msg))
                # spy: I am the spy
                msg = random.choice(SPY_REVEAL)
                output.append((game.spy, msg))

        return output

    def pregenerate_audio(self):
        """pre-generates audio for the game"""
        conversation = self.get_conversation()
        self.audio: list[int, np.ndarray] = None

    def render(self):
        game = self.game
        print(game.window)
        # render the round
