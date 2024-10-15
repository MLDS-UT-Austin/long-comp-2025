from simulation import *
from util import *

if __name__ == "__main__":
    # Select the NLP model to use ####################################################

    # 1: Use Together.ai for Llama and BERT
    # nlp = NLP(llm=Llama(), embedding=BERTTogether())

    # 2: Use Together.ai for Llama but run BERT locally
    # nlp = NLP(llm = Llama(), embedding=BERTLocal(batch_size=8))

    # 3: Output the same string for every question and return a 0 array for every embedding
    nlp = NLP(llm=DummyLLM(), embedding=DummyEmbedding())

    # Load agents from specific files ####################################################
    import_agents_from_files("submission.py")  # for you to run your agent
    import_agents_from_files("agents/MLDS 0/submission.py")  # for you to run your agent
    # import_agents_from_files("github classroom submissions/**/submission.py") # for us to run your agents

    # Run a single game ####################################################
    game = Game(player_names=["MLDS 0", "MLDS 0", "MLDS 0"], nlp=nlp, n_rounds=20)
    asyncio.run(game.play())
    print(game)
    print("Scores:", game.get_scores())
    print("Percent Right Votes:", game.get_percent_right_votes())
    print("Game Duration:", len(game.rounds))
    game.get_conversation().to_csv("conversation.csv", index=False)
    game.pregenerate_audio()
    game.render()
    game.save_audio("game.wav")

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
