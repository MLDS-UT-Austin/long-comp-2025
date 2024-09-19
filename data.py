from enum import Enum


class Location(Enum):
    AIRPLANE = "Airplane"
    BANK = "Bank"
    BEACH = "Beach"
    BROADWAY_THEATER = "Broadway Theater"
    CASINO = "Casino"
    CATHEDRAL = "Cathedral"
    CIRCUS_TENT = "Circus Tent"
    CORPORATE_PARTY = "Corporate Party"
    CRUSADER_ARMY = "Crusader Army"
    DAY_SPA = "Day Spa"
    EMBASSY = "Embassy"
    HOSPITAL = "Hospital"
    HOTEL = "Hotel"
    MILITARY_BASE = "Military Base"
    MOVIE_STUDIO = "Movie Studio"
    OCEAN_LINER = "Ocean Liner"
    PASSENGER_TRAIN = "Passenger Train"
    PIRATE_SHIP = "Pirate Ship"
    POLAR_STATION = "Polar Station"
    POLICE_STATION = "Police Station"
    RESTAURANT = "Restaurant"
    SCHOOL = "School"
    SERVICE_STATION = "Service Station"
    SPACE_STATION = "Space Station"
    SUBMARINE = "Submarine"
    SUPERMARKET = "Supermarket"
    UNIVERSITY = "University"


SPY_REVEAL_AND_GUESS = (
    "Muah ha ha! I was the spy all along! Was it the {location}?",
    "Jokes on you all! I was the spy! Was the {location}.",
    "You never suspected me, did you? I was right under your noses! Was it the {location}?",
    "Congratulations, you’ve played right into my hands. I was the spy! Was it the {location}?",
    "All your efforts were in vain. The spy was me and the location is the {location}!",
    "You were all so close, yet so far. I was the spy all along! Was it the {location}?",
    "The spy was me! I think it's the {location}.",
    "I’ve been pulling the strings from behind the scenes. I am the spy! Was it the {location}?",
)

SPY_GUESS_RIGHT_RESPONSE = (
    "You got us! That’s the right location!",
    "You got us! That’s right!",
    "We should have known it was you! You got it right!",
    "You got us! That’s the correct location!",
    "Ah, you got us! That’s the right location!",
)

SPY_GUESS_WRONG_RESPONSE = (
    "Nope! It was the {location}.",
    "No, it was the {location}.",
    "Close, but no! It was the {location}.",
    "Nope! It was the {location}. We win!",
)

ACCUSATION = (
    "I think it's player {spy}. Are you the spy?",
    "I accuse player {spy} of being the spy. Are you the spy?",
    "I suspect player {spy} of being the spy. Is it you?",
    "I have a feeling it's player {spy}. Are you the spy?",
    "I think player {spy} is the spy. Are you the spy?",
)

SPY_INDICTED_RESPONSE = (
    "Ah, you got me! I am the spy.",
    "You got me! I am the spy.",
    "You caught me! I am the spy.",
    "Guilty as charged! I am the spy.",
    "Yep, it was me!",
)

NON_SPY_INDICTED_RESPONSE = (
    "No, I am not the spy.",
    "You’re wrong! I am not the spy.",
    "I am not the spy.",
    "You’re mistaken! I am not the spy.",
    "Nope, not the spy.",
)

SPY_REVEAL = (
    "Muah ha ha! I was the spy all along!",
    "Jokes on you all! I was the spy!",
    "You never suspected me, did you? I was right under your noses!",
    "Muah ha ha! Y'all played right into my hands! I was the spy!",
    "All your efforts were in vain. I was the spy all along!",
    "You were all so close, yet so far. I was the spy all along!",
    "I’ve been pulling the strings from behind the scenes. I am the spy!",
)
