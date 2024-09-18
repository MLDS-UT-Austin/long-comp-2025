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


SPY_MONOLOUGES = (
    "Muah ha ha! I was the spy all along! Was it the {location}?",
    "Jokes on you all! I was the spy! Was the {location}.",
    "You never suspected me, did you? The spy was right under your noses! Was it the {location}?",
    "It’s been a pleasure watching you all struggle while I stayed hidden. Was it the {location}?",
    "Congratulations, you’ve played right into my hands. I was the spy! Was it the {location}?",
    "All your efforts were in vain. The spy was me and the location is the {location}!",
    "You were all so close, yet so far. I was the spy all along! Was it the {location}?",
    "The true mastermind has revealed themselves. The spy was me! I think it's the {location}.",
    "I’ve been pulling the strings from behind the scenes. I am the spy! Was it the {location}?",
)
