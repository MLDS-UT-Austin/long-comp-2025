import asyncio
import importlib
import os
import re
import sys
import time
from collections import Counter
from functools import wraps
from glob import glob
from os.path import dirname

import gtts  # type: ignore
import numpy as np
import parselmouth
from gtts import gTTS  # type: ignore
from parselmouth.praat import call
from together.error import RateLimitError

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


def count_votes(votes: list[int | None], n_players: int) -> int | None:
    """Used in game.py to count votes and determine the majority
    Args:
        votes (list[int  |  None]): the player that each player voted for, or None if they abstained
        n_players (int): the number of players in the game
    """
    counter = Counter(votes)
    if counter[None] >= n_players / 2:
        return None  # half or more abstained
    else:
        del counter[None]
        top2 = counter.most_common(2)  # ((player, count), (player, count))
        if len(top2) == 1 or top2[0][1] > top2[1][1]:
            return top2[0][0]
        else:
            return None  # tie


def import_agents_from_files(glob_pattern: str) -> None:
    """loads an agent from a file and adds it to the agent registry
    Correctly handles imports, prioritizing the working directory over the submission directory
    Args:
        file_path (str): the path to the agent's submission file
    """
    for file in glob(glob_pattern, recursive=True):
        # submitted agents will prioritize files in the working directory, then the submission directory
        # name = file
        name = f"agent_{dirname(file).replace('/', '_')}_{file.split('/')[-1].replace('.py', '')}"
        spec = importlib.util.spec_from_file_location(name, file)  # type: ignore
        module = importlib.util.module_from_spec(spec)  # type: ignore
        orig_path = sys.path
        sys.modules[name] = module
        sys.path.insert(1, dirname(file))
        spec.loader.exec_module(module)
        sys.path = orig_path


def rate_limit(requests_per_second: int):
    """A decorator to rate limit a function to a certain number of requests per second
    Args:
        requests_per_second (int): the number of requests per second
    """

    def decorator(func):
        last_time = 0
        request_lock = asyncio.Lock()

        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with request_lock:
                nonlocal last_time
                await asyncio.sleep(
                    last_time + (1 / requests_per_second) - time.monotonic()
                )
                last_time = time.monotonic()
            while True:
                wait_time = 0.01
                try:
                    return await func(*args, **kwargs)
                except RateLimitError:
                    # exponential backoff
                    await asyncio.sleep(wait_time)
                    wait_time *= 1.2

        return wrapper

    return decorator


VOICES = [
    ("en", "com.au"),
    ("en", "co.uk"),
    ("en", "us"),
    ("en", "co.in"),
    # ("en", "com.ng"),
    # ("fr", "fr"),
    # ("fr", "ca"),
    # ("pt", "com.br"),
    # ("pt", "pt"),
    # ("es", "com.mx"),
    # ("es", "es"),
]
PITCH_SHIFTS = [0.7, 0.85, 1.0, 1.15]


def text_to_speech(
    text, voice: tuple[str, str] = ("en", "com.au"), pitch_shift_factor: float = 1.0
) -> tuple[np.ndarray, int]:
    """_summary_

    Args:
        text (_type_): The text to convert to speech
        voice (tuple[str, str]): The voice to use for the speech, as a tuple of language and region
        pitch_shift_factor (float): The factor by which to shift the pitch

    Returns:
        tuple[np.ndarray, int]: The numpy array of the audio and the sample rate
    """
    lang, tld = voice

    # call api
    tts = gTTS(text=text, lang=lang, tld=tld)
    while True:
        try:
            tts.save(".temp.mp3")
            break
        except gtts.tts.gTTSError:
            continue

    # load into parselmouth object
    sound = parselmouth.Sound(".temp.mp3")
    os.remove(".temp.mp3")
    sr = int(sound.sampling_frequency)

    # Adjust pitch
    if pitch_shift_factor != 1.0:
        manipulation = call(sound, "To Manipulation", 0.01, 75, 600)
        pitch_tier = call(manipulation, "Extract pitch tier")
        call(
            pitch_tier,
            "Multiply frequencies",
            sound.xmin,
            sound.xmax,
            pitch_shift_factor,
        )
        call([pitch_tier, manipulation], "Replace pitch tier")
        sound = call(manipulation, "Get resynthesis (overlap-add)")

    x = np.squeeze(sound.values.T) * (2**15 - 1)
    x = x.astype(np.int16)

    return x, sr


if __name__ == "__main__":
    import pygame

    for ps in PITCH_SHIFTS:
        for voice in VOICES:
            print(f"Voice {voice}")
            text = "Hello, this is an AI voice generated from text."
            audio, sr = text_to_speech(text, voice, ps)

            pygame.mixer.pre_init(frequency=sr, channels=1, allowedchanges=0)
            pygame.init()
            sound = pygame.sndarray.make_sound(audio)
            sound.play()
            pygame.time.wait(int(sound.get_length() * 1000))
