import asyncio
import multiprocessing as mp
import os
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass
from glob import glob

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd

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
        player_classes = [
            random.sample(list(self.agent_registry.values()), self.team_size)
            for _ in range(self.n_games)
        ]
        tqdm_bar = tqdm(
            total=self.n_games * self.n_rounds, desc="Rounds", colour="green"
        )
        games = [Game(agents, self.llm, self.n_rounds) for agents in player_classes]
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

    def get_scores(self) -> pd.DataFrame:
        self.games, "must call run() first"
        df = pd.DataFrame(index=range(self.n_games), columns=self.agent_registry.keys())

        for i, game in enumerate(self.games):
            scores = game.get_scores()
            agent_names = [
                player_class.__name__ for player_class in game.player_classes
            ]
            df.loc[i, agent_names] = scores

        return df

    def visualize_scores(self, duration: int = 10, fps: int = 30):
        df = self.get_scores()
        df = df.cumsum(axis=0)

        plt.rcParams["toolbar"] = "None"

        fig, ax = plt.subplots()
        lines = [ax.plot([], [], lw=2, label=col)[0] for col in df.columns]
        ax.set_xlabel("Game")
        ax.set_ylabel("Cumulative Score")
        ax.set_title("Results")
        ax.legend(loc="upper left")

        def animate(animation_i):
            percent_complete = animation_i / (duration * fps)
            i = int(percent_complete * len(df))

            x = df.index[:i]
            for j, line in enumerate(lines):
                y = df[df.columns[j]][:i]
                line.set_data(x, y)

            if i > 0:
                ax.set_xlim(x.min(), x.max())
                ax.set_ylim(0, df.values[:i].max())

            return lines

        ani = animation.FuncAnimation(
            fig,
            animate,
            frames=duration * fps,
            interval=1000 / fps,
            repeat=False,
        )

        plt.show()


if __name__ == "__main__":
    sim = Simulation(DummyLLM())
    asyncio.run(sim.run())
    # sim.pickle_games()
    # print(sim.get_scores())
    sim.visualize_scores()
