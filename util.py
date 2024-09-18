from collections import Counter


def get_accused(votes: list[int | None], n_players: int):
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
