import importlib
import random
import string
import sys
from collections import Counter
from os.path import dirname


def count_votes(votes: list[int | None], n_players: int):
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


def import_agent_from_file(file_path: str) -> None:
    """load an agent from a file"""
    # submitted agents will prioritize files in the working directory, then the submission directory
    name = "".join(random.choice(string.ascii_letters) for _ in range(20))
    spec = importlib.util.spec_from_file_location(name, file_path)  # type: ignore
    module = importlib.util.module_from_spec(spec)  # type: ignore
    orig_path = sys.path
    sys.modules[name] = module
    sys.path.insert(1, dirname(file_path))
    spec.loader.exec_module(module)
    sys.path = orig_path
