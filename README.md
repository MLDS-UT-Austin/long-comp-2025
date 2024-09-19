<img src="media/readme_image.jpg" alt="spyfall" width="250" align="right" style="padding: 50px;"/>

# Long Competition 2025: Spyfall

This is the repository for the 2025 MLDS Long Competition where teams will be competing in a game of Spyfall. Teams will submit agents that can use LLMs and other NLP techniques to ask/answer questions and determine the spy/location. The goal of this Long Competition is to show your LLM prompting skills.

## Game Rules

**<u>Setup</u>**: Each game is played with 4-12 players. All players are given a scenario/location except for the spy.

**<u>Spy Objective</u>**: The spy must figure out the location without revealing their identity.

**<u>Non-Spy Objective</u>**: Players must figure out who the spy is.

**<u>Gameplay</u>**: Each game consists of a fixed number of rounds. In each round, the following happens:

1. **<u>Questioning</u>**: A random player starts by asking another player a question about the location. The player who answers the question will be the one to ask the question in the next round.
2. **<u>Questioning Analysis</u>**: All players are given time to analyze the question/answer.
3. **<u>Guessing</u>**: The spy may guess the location. This will end the game.
4. **<u>Accusation</u>**: Players may accuse another player of being the spy. Successfully indicting a player will end the game. For a player to be indicted, the following conditions must be met:
   * A majority of the players must accuse *any* player.
   * One player must get a plurality of the votes. If a tie occurs, nothing happens.

    Ex: If 2 players accuse player A, 1 player accuses player B, 1 player accuses player C, and 3 players do not vote, player A is successfully indicted.

5. **<u>Voting Analysis</u>**: Players can see who voted for who and are given time to analyze the votes.

**<u>The game ends when</u>**:

* The spy guesses the location.
* A player is indicted of being the spy.
* All rounds are completed.

**<u>Scoring</u>**:

* **<u>Spy Victory</u>**: The spy earns 2 points if no one is indicted of being the spy, 4 points if a non-spy player is indicted of being the spy, and 4 points if the spy stops the game and successfully guesses the location.
* **<u>Non-Spy Victory</u>**: Each non-spy player earns 1 point.

## Getting Started

**<u>Setting up the Code</u>**:

To run this project locally with conda, run the following commands:

``` bash
conda create -n long_comp python==3.10.13
conda activate long_comp
pip install -r requirements.txt
```

If you do not have conda installed, you can install the VSCode Extension `Python Environment Manager`, which should prompt you to install conda.

**<u>Setting up Your LLM API Key</u>**:

We will be using [together.ai](https://api.together.ai) which offers a $5 credit (~50M tokens) for new users, no credit card required.

To get your API key, click the link above to create an account. A pop-up will appear with your API key.

Next, create a file named `.env` in the root directory with the following text:

``` text
API_KEY = <your_api_key>
```

