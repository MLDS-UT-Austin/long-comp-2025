from agent import Agent, ExampleAgent
from game import Game


class MyAgent(Agent):
    pass


# Validate agent
MyAgent().validate()
# Play game

game = Game([MyAgent, ExampleAgent])
game.play()
game.score
