"""Enrich movies.csv with realistic film titles while strictly preserving IDs, years, and genres."""

import random
from pathlib import Path

import pandas as pd

# Real iconic movie titles categorized by primary genre
GENRE_TITLES = {
    "Action": [
        "The Dark Knight", "Mad Max: Fury Road", "Die Hard", "Gladiator", "Terminator 2: Judgment Day",
        "John Wick", "The Matrix", "Raiders of the Lost Ark", "Aliens", "Casino Royale",
        "Speed", "Léon: The Professional", "The Bourne Ultimatum", "Mission: Impossible - Fallout",
        "Kill Bill: Vol. 1", "Oldboy", "Crouching Tiger, Hidden Dragon", "Heat", "The Avengers",
        "Black Panther", "Edge of Tomorrow", "Top Gun: Maverick", "Ip Man", "The Raid: Redemption",
        "Skyfall", "RoboCop", "Baby Driver", "Predator", "First Blood", "Point Break"
    ],
    "Adventure": [
        "The Lord of the Rings: The Fellowship of the Ring", "Jurassic Park", "Star Wars: A New Hope",
        "Indiana Jones and the Last Crusade", "Pirates of the Caribbean: The Curse of the Black Pearl",
        "Cast Away", "Life of Pi", "The Revenant", "Stand by Me", "Lawrence of Arabia",
        "Back to the Future", "Avatar", "King Kong", "The Princess Bride", "The Goonies",
        "The Mummy", "Treasure Island", "Into the Wild", "127 Hours", "Everest",
        "Dune: Part One", "The Secret Life of Walter Mitty", "Jungle Cruise", "Stardust"
    ],
    "Animation": [
        "Toy Story", "Spirited Away", "The Lion King", "WALL-E", "Spider-Man: Into the Spider-Verse",
        "Up", "Finding Nemo", "Monsters, Inc.", "Coco", "Ratatouille", "Princess Mononoke",
        "How to Train Your Dragon", "Inside Out", "The Incredibles", "Shrek", "Aladdin",
        "Beauty and the Beast", "Zootopia", "Grave of the Fireflies", "My Neighbor Totoro",
        "Klaus", "Soul", "Your Name", "Howl's Moving Castle", "The Iron Giant", "Fantastic Mr. Fox"
    ],
    "Comedy": [
        "The Big Lebowski", "Superbad", "Groundhog Day", "Monty Python and the Holy Grail",
        "The Truman Show", "Shaun of the Dead", "Hot Fuzz", "Airplane!", "Ferris Bueller's Day Off",
        "Anchorman: The Legend of Ron Burgundy", "Step Brothers", "Tropic Thunder", "Zoolander",
        "Office Space", "School of Rock", "Midnight in Paris", "Borat", "Dumb and Dumber",
        "The Hangover", "Mean Girls", "Clerks", "Knives Out", "What We Do in the Shadows", "Rushmore"
    ],
    "Crime": [
        "The Godfather", "Pulp Fiction", "Goodfellas", "The Departed", "Scarface",
        "The Untouchables", "Casino", "Fargo", "City of God", "No Country for Old Men",
        "Reservoir Dogs", "Chinatown", "The Usual Suspects", "Snatch", "L.A. Confidential",
        "Sicario", "Training Day", "American Gangster", "Drive", "Prisoners", "Catch Me If You Can",
        "The Irishman", "Road to Perdition", "Public Enemies", "Donnie Brasco"
    ],
    "Documentary": [
        "Free Solo", "Won't You Be My Neighbor?", "March of the Penguins", "Jiro Dreams of Sushi",
        "Man on Wire", "The Cove", "Senna", "13th", "Apollo 11", "Searching for Sugar Man",
        "Grizzly Man", "Bowling for Columbine", "They Shall Not Grow Old", "Fyre", "Wild Wild Country",
        "The Social Dilemma", "O.J.: Made in America", "Amy", "Icarus", "The Dawn Wall"
    ],
    "Drama": [
        "The Shawshank Redemption", "Forrest Gump", "Schindler's List", "Fight Club", "12 Angry Men",
        "One Flew Over the Cuckoo's Nest", "Whiplash", "A Beautiful Mind", "The Prestige",
        "There Will Be Blood", "The Social Network", "Manchester by the Sea", "Birdman",
        "Moonlight", "Parasite", "Dead Poets Society", "American Beauty", "Good Will Hunting",
        "The Green Mile", "Spotlight", "Room", "The Pianist", "Boyhood", "Babel", "Drive My Car"
    ],
    "Fantasy": [
        "The Lord of the Rings: The Return of the King", "Harry Potter and the Prisoner of Azkaban",
        "Pan's Labyrinth", "The Wizard of Oz", "Edward Scissorhands", "Big Fish", "Stardust",
        "The Shape of Water", "Coraline", "Bridge to Terabithia", "The Chronicles of Narnia",
        "The Golden Compass", "Labyrinth", "The Dark Crystal", "Percy Jackson", "Eragon",
        "A Monster Calls", "Alice in Wonderland", "Willow", "The NeverEnding Story"
    ],
    "Film-Noir": [
        "The Third Man", "Double Indemnity", "The Maltese Falcon", "Sunset Boulevard", "Touch of Evil",
        "The Big Sleep", "Strangers on a Train", "Laura", "Out of the Past", "Shadow of a Doubt",
        "In a Lonely Place", "Gilda", "The Night of the Hunter", "Sweet Smell of Success", "Kiss Me Deadly",
        "Night and the City", "The Killing", "Notorious", "The Postman Always Rings Twice", "Key Largo"
    ],
    "Horror": [
        "The Shining", "The Silence of the Lambs", "Alien", "Psycho", "The Thing",
        "Hereditary", "Get Out", "A Quiet Place", "Halloween", "The Exorcist",
        "The Texas Chain Saw Massacre", "Scream", "The Conjuring", "Midsommar", "It Follows",
        "The Babadook", "28 Days Later", "The Cabin in the Woods", "Saw", "The Witch",
        "Insidious", "Sinister", "Poltergeist", "Rosemary's Baby", "Evil Dead II"
    ],
    "Musical": [
        "La La Land", "Singin' in the Rain", "The Sound of Music", "West Side Story", "Chicago",
        "The Greatest Showman", "Moulin Rouge!", "Les Misérables", "Mary Poppins", "Cabaret",
        "Grease", "Fiddler on the Roof", "Hairspray", "Sweeney Todd", "Mamma Mia!",
        "Rocketman", "Bohemian Rhapsody", "Phantom of the Opera", "Oliver!", "A Star Is Born"
    ],
    "Mystery": [
        "Memento", "Se7en", "Zodiac", "Shutter Island", "The Prestige", "The Girl with the Dragon Tattoo",
        "Rear Window", "Vertigo", "Prisoners", "Gone Girl", "Knives Out", "Glass Onion",
        "The Da Vinci Code", "Source Code", "Identity", "Mystic River", "Primal Fear",
        "Insomnia", "Mulholland Drive", "The Sixth Sense", "A Haunting in Venice", "Sherlock Holmes"
    ],
    "Romance": [
        "Titanic", "Casablanca", "Eternal Sunshine of the Spotless Mind", "Before Sunrise", "Before Sunset",
        "The Notebook", "Pride & Prejudice", "Amélie", "500 Days of Summer", "About Time",
        "Her", "Brokeback Mountain", "Atonement", "The Fault in Our Stars", "Crazy Rich Asians",
        "Silver Linings Playbook", "Roman Holiday", "When Harry Met Sally...", "Portrait of a Lady on Fire",
        "Call Me by Your Name", "Carol", "Little Women", "Me Before You", "Brooklyn"
    ],
    "Sci-Fi": [
        "Interstellar", "Inception", "The Matrix", "2001: A Space Odyssey", "Blade Runner 2049",
        "Arrival", "Dune", "Star Wars: The Empire Strikes Back", "The Martian", "Ex Machina",
        "Minority Report", "Children of Men", "District 9", "Gravity", "Looper",
        "Gattaca", "Contact", "Dark City", "Snowpiercer", "Annihilation",
        "Solaris", "The Fifth Element", "Tenet", "E.T. the Extra-Terrestrial", "Close Encounters of the Third Kind"
    ],
    "Thriller": [
        "Parasite", "Joker", "Nightcrawler", "Gone Girl", "Drive", "Black Swan", "Sicario",
        "The Fugitive", "Misery", "Cape Fear", "Fatal Attraction", "The Game", "Collateral",
        "Panic Room", "Run Lola Run", "Speed", "Don't Breathe", "Split", "Searching",
        "A Simple Plan", "Uncut Gems", "Wind River", "Green Room", "The Invisible Man"
    ],
    "War": [
        "Saving Private Ryan", "Apocalypse Now", "Full Metal Jacket", "1917", "Dunkirk",
        "Platoon", "The Hurt Locker", "The Thin Red Line", "Hacksaw Ridge", "Inglourious Basterds",
        "Black Hawk Down", "Paths of Glory", "The Bridge on the River Kwai", "Letters from Iwo Jima",
        "Das Boot", "Enemy at the Gates", "Fury", "Lone Survivor", "We Were Soldiers"
    ],
    "Western": [
        "The Good, the Bad and the Ugly", "Unforgiven", "Once Upon a Time in the West", "Django Unchained",
        "True Grit", "The Searchers", "Tombstone", "High Noon", "3:10 to Yuma", "The Hateful Eight",
        "Butch Cassidy and the Sundance Kid", "A Fistful of Dollars", "For a Few Dollars More",
        "The Magnificent Seven", "Dances with Wolves", "Bone Tomahawk", "The Outlaw Josey Wales",
        "Pale Rider", "Open Range", "Hell or High Water"
    ],
}


def enrich_movies_dataset(movies_path: Path):
    df = pd.read_csv(movies_path)
    used_titles = set()
    new_titles = []

    # Deterministic mapping
    random.seed(42)

    for _, row in df.iterrows():
        mid = row["movieId"]
        genres = str(row["genres"]).split("|")
        primary_genre = genres[0] if genres else "Drama"
        candidates = GENRE_TITLES.get(primary_genre, GENRE_TITLES["Drama"])

        # Extract year from existing title if present (e.g. "Movie 1 (1977)")
        orig_title = str(row["title"])
        year_str = ""
        if "(" in orig_title and orig_title.endswith(")"):
            year_str = orig_title[orig_title.rfind("("):]

        # Select candidate
        available = [t for t in candidates if t not in used_titles]
        if available:
            base_name = available[0]
            used_titles.add(base_name)
            title = f"{base_name} {year_str}".strip()
        else:
            # Fallback with variation (e.g. "The Dark Knight: Legacy (1977)")
            base_name = random.choice(candidates)
            variant = random.choice(["Chapter II", "Chronicles", "Origins", "Legacy", "Redemption", "Rising", "Reckoning", "Returns", "Vol. 2", "Final Cut"])
            candidate_title = f"{base_name}: {variant}"
            if candidate_title not in used_titles:
                used_titles.add(candidate_title)
                title = f"{candidate_title} {year_str}".strip()
            else:
                title = f"{base_name} #{mid} {year_str}".strip()

        new_titles.append(title)

    df["title"] = new_titles
    df.to_csv(movies_path, index=False)
    print(f"Successfully enriched {len(df)} movies in {movies_path}")
    print("Sample enriched entries:")
    print(df.head(10)[["movieId", "title", "genres"]])


if __name__ == "__main__":
    enrich_movies_dataset(Path("data/raw/ml-latest-small/movies.csv"))
