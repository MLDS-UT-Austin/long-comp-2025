import asyncio
import multiprocessing as mp
import os
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass
from glob import glob

from agent import *
from game import *
from llm import *
from util import *


@dataclass
class Simulation:
    llm: LLM
    submission_paths: str = "agents/**/submission.py"
    gave_save_dir: str = "games/simulation0"
    n_games: int = 100
    team_size: int = 2
    n_rounds: int = 20

    def __post_init__(self):
        os.makedirs(self.gave_save_dir, exist_ok=True)
        for file in glob(self.submission_paths, recursive=True):
            import_agent_from_file(file)
        self.agent_registry = AGENT_REGISTRY.copy()
        for name, agent_class in self.agent_registry.items():
            try:
                agent_class.validate()
            except Exception as e:
                print(f"Agent {name} failed validation: {e}")
                raise e

    async def run(self):
        agent_classes = [
            random.sample(list(self.agent_registry.values()), self.team_size)
            for _ in range(self.n_games)
        ]
        tqdm_bar = tqdm(
            total=self.n_games * self.n_rounds, desc="Rounds", colour="green"
        )
        games = [Game(agents, self.llm, self.n_rounds) for agents in agent_classes]
        for game in games:
            game.tqdm_bar = tqdm_bar

        await asyncio.gather(*[game.play() for game in games])

        for game in games:
            game.tqdm_bar = None
        tqdm_bar.close()

        self.games = games

    def pickle_games(self):
        self.games, "must call run() first"

        for i, game in enumerate(self.games):
            with open(f"{self.gave_save_dir}/game_{i}.pkl", "wb") as f:
                pickle.dump(game, f)

    def load_games(self):
        games = []

        for path in glob(f"{self.gave_save_dir}/*.pkl"):
            with open(path, "rb") as f:
                games.append(pickle.load(f))

        self.games = games

    def get_scores(self) -> dict[str, list[int]]:
        self.games, "must call run() first"
        output = defaultdict(list)
        for game in self.games:
            scores = game.get_scores()
            for name, score in scores.items():
                output[name].append(score)
        return output


if __name__ == "__main__":
    sim = Simulation(DummyLLM())
    asyncio.run(sim.run())
    sim.pickle_games()
    print(sim.get_scores())
