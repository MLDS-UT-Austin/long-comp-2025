import asyncio
from dataclasses import dataclass

from agent import Agent


@dataclass
class Simulation:
    agent_classes: list[type[Agent]]

    pass

@dataclass
class Game:
    agent_classes: list[type[Agent]]
    n_rounds: int = 20

    def __post_init__(self):
        # pick location
        # pick spy
        self.players: list[Agent] = []
        self.location = None
        self.spy = None
        self.current_player = 0
        self.conversation = None
        pass

    def play(self):
        for _ in range(self.n_rounds):
            self.round()
        # final vote

    def round(self):
        # ask questions
        answerer, question = self.players[self.current_player].ask_question()
        answer = self.players[answerer].answer_question(question)
        self.conversation = self.converation + "f Player {self.current_player} asked player{answerer}: {question}"
        self.converation = self.conversation + "f Player {answerer} responded {answer}."
        for player in self.players:
            player.analyze_response(self.current_player, question, answerer, answer)

        guess = self.players[self.spy].guess_location()
        if guess == None:
            pass
        if guess == self.location:
            self.converation = self.conversation + "f A majority of people voted for {self.location}. The spy was {}"
            print("Spy wins")
            return
        else:
            print("Spy loses")

        self.current_player = answerer

        votes = [player.accuse_player() for player in self.players]
        majority = max(votes)
        if majority == None:
            pass
        else:
            print(f"Player {majority} was accused")


        # answer questions
        # analyze responses
        # guess location
        # accuse player
        # analyze voting
        pass

    def pregenerate_audio(self):
        """pre-generates audio for the game"""
        self.audio = 

    def render():
        pass

