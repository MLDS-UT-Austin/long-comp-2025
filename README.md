# Long Competition 2025: Spyfall

This is the repository for the 2025 MLDS Long Compeition where teams will be competing in a game of Spyfall. Teams will submit agents that can use LLMs and other MLP techniques to ask/answer questions and determine the spy/location. The goal of this Long Competition is show your LLM prompting skills.

## Game Rules

**Setup:** Each game is player with 4-12 players. All players are given a scenario/location except for the spy.

**Spy Objective:** The spy must figure out the location without revealing their identity.

**Non-Spy Objective:** Players must figure out who the spy is.

### Gameplay

Each game consists of a fixed number of rounds. In each round, the following happens:

1. **Questioning:** A random player starts by asking another player a question about the location. The player who answers the question will be the one to ask the question the next round.
2. **Questioning Analysis:** All players are given time to analyze the question/answer.
3. **Guessing** The spy may guess the location. This will end the game.
4. **Accusation:** Players may accuse another player of being the spy. If a majority of the players chose to accuse, the plurality wins and the game ends. Nothing happens if there is a tie.
   Ex: If 2 players accuse player A, 1 player accuses player B, 1 player accuses player C, and 3 players do not vote, player A is successfully accused.
5. **Voting Analysis:** Players can see who voted for who and are given time to analyze the votes.

### The game ends when:

- The spy guesses the location.
- A player is successfully accused of being the spy.
- All rounds are completed.

### Scoring

- **Spy Victory:** The spy earns 2 points if no one is successfully accused of being the spy, 4 points if a non-spy player is successfully accused of being a spy, and 4 points if the spy stops the game and successfully guesses the location.
- **Non-Spy Victory:** Each non-spy player earns 1 point.

### Getting Started

To run this project locally with conda, run the following commands:

``` bash
conda create -n long_comp python==3.10.13
conda activate long_comp
pip install tqdm python-dotenv numpy tiktoken
```

If you do not have conda, you can install the VSCode Extension `Python Environment Manager`, then it should prompt you to install conda.

