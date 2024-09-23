import importlib
import random
import re
import string
import sys
from collections import Counter
from glob import glob
from os.path import dirname

from data import *


def redact(text: str, location: Location, redacted_text: str = "<REDACTED>") -> str:
    """Can optionally by used by agents to redact text based on the location
    This can be useful to prevent the LLM from giving away the location
    Note: this is not called in game.py
    Args:
        text (str): text to redact
        location (Location): the location to redact
        redacted_text (str, optional): what to replace the redacted text with
    """
    for word in redaction_dict[location]:
        text = re.sub(rf"{word}", redacted_text, text, flags=re.IGNORECASE)
    return text


def count_votes(votes: list[int | None], n_players: int):
    """Used in game.py to count votes and determine the majority
    Args:
        votes (list[int  |  None]): the player that each player voted for, or None if they abstained
        n_players (int): the number of players in the game
    """
    counter = Counter(votes)
    if counter[None] >= n_players / 2:
        majority = None  # half or more abstained
    else:
        del counter[None]
        top2 = counter.most_common(2)  # ((player, count), (player, count))
        if len(top2) == 1 or top2[0][1] > top2[1][1]:
            majority = top2[0][0]
        else:
            majority = None  # tie


def import_agents_from_files(glob_pattern: str) -> None:
    """loads an agent from a file and adds it to the agent registry
    Correctly handles imports, prioritizing the working directory over the submission directory
    Args:
        file_path (str): the path to the agent's submission file
    """
    for file in glob(glob_pattern, recursive=True):
        # submitted agents will prioritize files in the working directory, then the submission directory
        name = "".join(random.choice(string.ascii_letters) for _ in range(20))
        spec = importlib.util.spec_from_file_location(name, file)  # type: ignore
        module = importlib.util.module_from_spec(spec)  # type: ignore
        orig_path = sys.path
        sys.modules[name] = module
        sys.path.insert(1, dirname(file))
        spec.loader.exec_module(module)
        sys.path = orig_path
