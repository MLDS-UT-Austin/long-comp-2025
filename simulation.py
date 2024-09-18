import asyncio
import multiprocessing as mp
import os
import pickle
import random
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
    team_size: int = 10
    n_rounds: int = 20

    def __post_init__(self):
        os.makedirs(self.gave_save_dir, exist_ok=True)
        for file in glob(self.submission_paths, recursive=True):
            import_agent_from_file(file)
        for name, agent_class in AGENT_REGISTRY.items():
            try:
                agent_class.validate()
            except Exception as e:
                print(f"Agent {name} failed validation: {e}")
                raise e

        # FIXME: temporary fix for not having enough agents
        AGENT_REGISTRY.update(
            {f"RandomAgent{i}": AGENT_REGISTRY["TeamNameHere"] for i in range(10)}
        )

    async def run(self):
        players = [
            random.sample(list(AGENT_REGISTRY.values()), self.team_size)
            for _ in range(self.n_games)
        ]
        tqdm_bar = tqdm(
            total=self.n_games * self.n_rounds, desc="Rounds", colour="green"
        )
        games = [Game(agents, self.llm, self.n_rounds, tqdm_bar) for agents in players]

        await asyncio.gather(*[game.play() for game in games])

        tqdm_bar.close()
        self.games = games

    def pickle_games(self):
        self.games, "must call run() first"

        def fun(i, game):
            with open(f"{self.gave_save_dir}/game_{i}.pkl", "wb") as f:
                pickle.dump(game, f)

        with mp.Pool(8) as pool:
            results = pool.map(fun, self.games)

    def load_games(self):
        games = [] # TODO: check this
        for file in glob(f"{self.gave_save_dir}/game_*.pkl"):
            with open(file, "rb") as f:
                game = pickle.load(f)
            games.append(game)
        self.games = games

    def get_scores(self) -> np.ndarray:
        self.games, "must call run() first"
        scores = np.array((game.get_scores() for game in self.games))
        return scores


if __name__ == "__main__":
    sim = Simulation(DummyLLM())
    asyncio.run(sim.run())
