import asyncio
import random
from collections import Counter
from dataclasses import dataclass
from enum import Enum

from agent import Agent
from data import *
from llm import *
from util import *


class GameState(Enum):
    RUNNING = "ongoing"
    SPY_GUESSED_RIGHT = "spy guessed right"
    SPY_GUESSED_WRONG = "spy guessed wrong"
    SPY_ACCUSED = "spy accused"
    NON_SPY_ACCUSED = "non-spy accused"
    NO_ONE_ACCUSED = "no one accused"


class Game:
    location: Location
    spy: int
    questioner: int
    players: list[Agent]
    player_llms: list[LLMTokenCounterWrapper]
    game_state: GameState

    spy_scoring = {
        GameState.RUNNING: 0,
        GameState.SPY_GUESSED_RIGHT: 4,
        GameState.SPY_GUESSED_WRONG: 0,
        GameState.SPY_ACCUSED: 0,
        GameState.NON_SPY_ACCUSED: 4,
        GameState.NO_ONE_ACCUSED: 2,
    }
    player_scoring = {
        GameState.RUNNING: 0,
        GameState.SPY_GUESSED_RIGHT: 0,
        GameState.SPY_GUESSED_WRONG: 1,
        GameState.SPY_ACCUSED: 1,
        GameState.NON_SPY_ACCUSED: 0,
        GameState.NO_ONE_ACCUSED: 0,
    }

    def __init__(
        self,
        agent_classes: list[type[Agent]],
        n_rounds: int = 20,
    ):
        self.n_rounds = n_rounds
        n_players = self.n_players = len(agent_classes)

        self.location = random.choice(list(Location))
        self.spy = random.randint(0, n_players - 1)
        self.questioner = random.randint(0, n_players - 1)
        self.players = []
        self.player_llms = []
        self.game_state = GameState.RUNNING

        llm = OpenAILLM()

        for i, agent_class in enumerate(agent_classes):
            player_llm = LLMTokenCounterWrapper(llm)
            given_location = self.location if i != self.spy else None
            agent = agent_class(
                given_location, n_players, n_rounds, llm=LLMProxy(player_llm)
            )
            self.players.append(agent)
            self.player_llms.append(player_llm)

        self.povs = [list(range(1, n_players))] * n_players
        for i, pov in enumerate(self.povs):
            random.shuffle(pov)
            pov.insert(i, 0)
        self.r_povs = [[0] * (n_players - 1)] * n_players
        for i in range(n_players):
            for player, player_w_pov in enumerate(self.povs[i]):
                self.r_povs[i][player_w_pov] = player

        self.rounds: list[Round] = []

    def add_pov(self, player: int, pov: int):
        return self.povs[pov][player]

    def reverse_pov(self, player: int, pov: int):
        return self.r_povs[pov][player]

    async def play(self):
        for _ in range(self.n_rounds):
            round = Round(self)
            await round.play()
            self.rounds.append(round)
            if self.game_state != GameState.RUNNING:
                return
        self.game_state = GameState.NO_ONE_ACCUSED

    def pregenerate_audio(self):
        """pre-generates audio for the game"""
        self.audio = None

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
    accused: int | None

    def __init__(self, game: Game):
        self.game = game

    async def play(self):
        game = self.game
        for llm in game.player_llms:
            llm.reset_token_counter()
        questioner = self.questioner = game.questioner

        # TODO: should we increase the token count for the questioner and answerer?
        answerer, question = await game.players[questioner].ask_question()
        answerer = game.reverse_pov(answerer, pov=questioner)
        answer = await game.players[answerer].answer_question(question)
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
        if guess == game.location:
            game.game_state = GameState.SPY_GUESSED_RIGHT
            return
        elif guess != None:
            game.game_state = GameState.SPY_GUESSED_WRONG
            return

        # player voting
        votes = self.player_votes = await asyncio.gather(
            [player.accuse_player() for player in game.players]
        )

        for i, vote in enumerate(votes):
            if vote is not None:
                votes[i] = game.reverse_pov(vote, pov=i)

        accused = self.accused = get_accused(votes, game.n_players)
        if accused == game.spy:
            game.game_state = GameState.SPY_ACCUSED
        elif accused is not None:
            game.game_state = GameState.NON_SPY_ACCUSED
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
            msg = random.choice(SPY_GUESS_QUESTION).format(
                location=self.spy_guess.value
            )
            output.append((game.spy, msg))
            responder = random.choice(list(set(range(game.n_players)) - {game.spy}))
            if game.game_state == GameState.SPY_GUESSED_RIGHT:
                # TODO
                # random nonspy: yes that is right
                msg = random.choice(SPY_GUESS_RIGHT_RESPONSE)
            else:
                # TODO
                # random nonspy: no it was the {location}
                msg = random.choice(
                    SPY_GUESS_WRONG_RESPONSE.format(location=game.location.value)
                )
            output.append((responder, msg))

        # write logic for accusations here if there is a majority
        if self.accused is not None:
            # TODO
            # one of the accusers: "I think it's player {spy} are you the spy?"
            msg = random.choice(SPY_ACCUSED_QUESTION).format(spy=game.spy + 1)
            if game.game_state == GameState.SPY_ACCUSED:
                # TODO
                # spy: I am the spy
                msg = random.choice(SPY_ACCUSED_RIGHT_RESPONSE)
            else:
                # TODO
                # accused: No, I am not the spy
                msg = random.choice(SPY_ACCUSED_WRONG_ACCUSED_RESPONSE)
                # spy: I am the spy
                msg = random.choice(SPY_ACCUSED_WRONG_SPY_RESPONSE)

        return output

        # self.conversation = (
        #     self.converation
        #     + "f Player {self.current_player} asked player{answerer}: {question}"
        # )
        # self.converation = self.conversation + "f Player {answerer} responded {answer}."
        # self.converation = (
        #     self.conversation
        #     + "f A majority of people voted for {self.location}. The spy was {}"
        # )

    def render(self):
        game = self.game
        print(game.window)
        # render the round


class Simulation:
    agent_classes: list[type[Agent]]

    pass
