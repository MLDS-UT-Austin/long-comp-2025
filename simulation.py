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
import pandas as pd  # type: ignore

from agent import *
from game import *
from nlp import *
from util import *


@dataclass
class Simulation:
    """A class to run multiple games concurrently and save/analyze the results"""

    nlp: NLP
    agent_names: list[str] | None = None
    gave_save_dir: str = "games/simulation0"
    team_size: int = 2
    n_rounds: int = 20

    def __post_init__(self):
        if self.agent_names is None:
            self.agent_names = list(AGENT_REGISTRY.keys())
        self.games: list[Game] = []

    def validate_agents(self):
        for name in set(self.agent_names):
            try:
                AGENT_REGISTRY[name].validate()
            except Exception as e:
                print(f"Agent {name} failed validation: {e}")
                raise e

    async def run(self, n_games: int = 1):
        """Run multiple games in parallel and adds the results to self.games
        Args:
            n_games (int, optional): number of games to run
            agent_names (list[str] | None, optional): names of agent classes to use
        """
        assert self.agent_names is not None
        # Randomly sample agent classes to play in each game
        sampled_agent_names = [
            random.sample(self.agent_names, self.team_size) for _ in range(n_games)
        ]

        # Set up progress bar
        tqdm_bar = tqdm(total=n_games * self.n_rounds, desc="Rounds", colour="green")

        # Run games concurrently
        games = [
            Game(self.nlp, agents, self.n_rounds) for agents in sampled_agent_names
        ]
        for game in games:
            game.tqdm_bar = tqdm_bar

        await asyncio.gather(*[game.play() for game in games])

        for game in games:
            game.tqdm_bar = None
        tqdm_bar.close()

        self.games.extend(games)

    def pickle_games(self):
        """Saves all games to self.gave_save_dir as pickle files"""
        assert len(self.games) > 0, "must call run() or load_games() first"
        os.makedirs(self.gave_save_dir, exist_ok=True)

        for i, game in enumerate(
            tqdm(self.games, desc="Pickling games", colour="green")
        ):
            with open(f"{self.gave_save_dir}/game_{i}.pkl", "wb") as f:
                pickle.dump(game, f)

    def load_games(self):
        """Loads all games from self.gave_save_dir"""
        games = []

        for path in tqdm(
            glob(f"{self.gave_save_dir}/*.pkl"), desc="Loading games", colour="green"
        ):
            with open(path, "rb") as f:
                games.append(pickle.load(f))

        self.games.extend(games)

    def get_scores(self) -> pd.DataFrame:
        """Get the scores of all agents in all games
        Returns:
            pd.DataFrame: Pandas DataFrame with the scores
                columns: agent names
                index: game number
                values: score or np.nan if the agent was not in the game
        """
        assert len(self.games) > 0, "must call run() or load_games() first"
        df = pd.DataFrame(index=range(len(self.games)), columns=self.agent_names)

        for i, game in enumerate(self.games):
            game_scores = game.get_scores()
            assert game_scores.index.isin(df.columns).all()
            df.loc[i] = game_scores

        return df

    def get_percent_right_votes(self) -> pd.DataFrame:
        """Get the percentage of right votes of all agents in all games
        Returns:
            pd.DataFrame: Pandas DataFrame with the percentage of right votes
                columns: agent names
                index: game number
                values: percentage of right votes or np.nan if the agent was not in the game
        """
        assert len(self.games) > 0, "must call run() or load_games() first"
        df = pd.DataFrame(index=range(len(self.games)), columns=self.agent_names)

        for i, game in enumerate(self.games):
            game_scores = game.get_percent_right_votes()
            assert game_scores.index.isin(df.columns).all()
            df.loc[i] = game_scores

        return df

    def get_game_endtype_freq(self) -> pd.Series:
        """Get the frequency of game end types in all games
        Returns:
            pd.Series: Pandas Series with the frequency of each game end type
                index: game end type
                values: frequency
        """
        c = Counter(g.game_state for g in self.games)
        return pd.Series(c)

    def get_game_duration_freq(self) -> pd.Series:
        """Get the frequency of game durations in all games
        Returns:
            pd.Series: Pandas Series with the frequency of each game duration
                index: game duration
                values: frequency
        """
        c = Counter(len(g.rounds) for g in self.games)
        return pd.Series(c)

    def _get_animation(
        self, duration: int = 10, fps: int = 30
    ) -> animation.FuncAnimation:
        df = self.get_scores()

        for col in df.columns:
            # Use expanding mean times index instead of cumsum to correctly handle NaNs
            df[col] = df[col].expanding().mean() * (df.index + 1)
            df[col] = df[col].fillna(0)

        plt.rcParams["toolbar"] = "None"

        fig, ax = plt.subplots()
        lines = [ax.plot([], [], lw=2, label=col)[0] for col in df.columns]
        ax.set_xlabel("Game")
        ax.set_ylabel("Cumulative Score")
        ax.set_title("Results")
        ax.legend(loc="upper left")

        def animate(animation_i):
            percent_complete = animation_i / (duration * fps)

            # adjust growth rate here to be non-linear if desired
            i = int(percent_complete * len(df))

            x = df.index[:i]
            for j, line in enumerate(lines):
                y = df[df.columns[j]][:i]
                line.set_data(x, y)

            # scale graph axes
            if i > 0:
                ax.set_xlim(0, max(1, x.max()))
                ax.set_ylim(0, max(1, df.values[:i].max()))

            return lines

        ani = animation.FuncAnimation(
            fig,
            animate,
            frames=duration * fps,
            interval=1000 / fps,
            repeat=False,
        )

        return ani

    def visualize_scores(self, duration: int = 10, fps: int = 30):
        """Plays visualization of the scores of all agents over time"""
        ani = self._get_animation(duration, fps)

        plt.show()

    def save_visualization(self, filepath, duration: int = 10, fps: int = 30):
        """Saves visualization of the scores of all agents over time to a file"""
        ani = self._get_animation(duration, fps)

        ani.save(filepath, writer="ffmpeg")


if __name__ == "__main__":
    # Select the NLP model to use ####################################################

    # 1: Use Together.ai for Llama and BERT
    # nlp = NLP(llm = Llama(), embedding=BERTTogether())

    # 2: Use Together.ai for Llama but run BERT locally
    # nlp = NLP(llm = Llama(), embedding=BERTLocal(batch_size=8))

    # 3: Output the same string for every question and return a 0 array for every embedding
    nlp = NLP(llm=DummyLLM(), embedding=DummyEmbedding())

    # Load agents from specific files ####################################################
    import_agents_from_files("submission.py")  # for you to run your agent
    import_agents_from_files("agents/MLDS 0/submission.py")  # for you to run your agent
    # import_agents_from_files("github classroom submissions/**/submission.py") # for us to run your agents

    # Run a single game ####################################################
    game = Game(player_names=["Example Agent", "MLDS 0"], nlp=nlp, n_rounds=20)
    asyncio.run(game.play())
    print("Scores:", game.get_scores())
    print("Percent Right Votes:", game.get_percent_right_votes())
    print("Game End Type:", game.game_state)
    print("Game Duration:", len(game.rounds))
    game.render()

    # Run multiple games with randomly sampled agents ####################################################
    sim = Simulation(nlp, agent_names=["Example Agent", "MLDS 0"])
    sim.validate_agents()
    asyncio.run(sim.run(n_games=10))
    # average scores of all agents
    print("Average Scores:", sim.get_scores().mean(axis=0))
    print("Average Percent Right Votes:", sim.get_percent_right_votes().mean(axis=0))
    print("Game End Type Frequency:", sim.get_game_endtype_freq())
    print("Game Duration Frequency:", sim.get_game_duration_freq())
    sim.visualize_scores()
    sim.save_visualization("simulation.mp4")
    sim.pickle_games()

    # Demo at Long Comp Intro Day
    game = Game(nlp=nlp, n_rounds=20)
    asyncio.run(game.play())
    game.render()

    # Long Comp Day - Show teams over time
    sim = Simulation(nlp)
    sim.validate_agents()
    asyncio.run(sim.run())
    sim.pickle_games()
    # average scores of all agents
    print("Average Scores:", sim.get_scores().mean(axis=0))
    sim.visualize_scores()
    sim.save_visualization("simulation.mp4")

    # Show final teams
    finalists = ["Example Agent", "Team Name Here"]
    game = Game(player_names=finalists, nlp=nlp, n_rounds=20)
    asyncio.run(game.play())
    game.render()
