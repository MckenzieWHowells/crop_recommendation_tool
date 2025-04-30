import random

# Sample dataset
catalogue = [
    {"title": "Stranger Things", "genre": "sci-fi", "mood": "exciting", "platform": "Netflix"},
    {"title": "The Crown", "genre": "drama", "mood": "serious", "platform": "Netflix"},
    {"title": "Brooklyn Nine-Nine", "genre": "comedy", "mood": "light", "platform": "Peacock"},
    {"title": "The Mandalorian", "genre": "sci-fi", "mood": "exciting", "platform": "Disney+"},
    {"title": "Fleabag", "genre": "comedy", "mood": "thoughtful", "platform": "Amazon Prime"},
    {"title": "Planet Earth", "genre": "documentary", "mood": "calm", "platform": "Netflix"},
    {"title": "Breaking Bad", "genre": "drama", "mood": "intense", "platform": "Netflix"},
    {"title": "The Office", "genre": "comedy", "mood": "light", "platform": "Peacock"},
]

# Input collection
print("Welcome to the Watch Recommendation Engine!\n")
genre = input("Choose a genre (e.g., comedy, drama, sci-fi, documentary): ").strip().lower()
mood = input("What mood are you in? (e.g., light, exciting, calm, serious, intense): ").strip().lower()
platform = input("Preferred streaming platform (e.g., Netflix, Disney+, Amazon Prime, Peacock): ").strip()

# Filtering logic
recommendations = [
    item for item in catalogue
    if item["genre"] == genre and item["mood"] == mood and item["platform"].lower() == platform.lower()
]

# Result
if recommendations:
    choice = random.choice(recommendations)
    print(f"\n🎬 You should watch: **{choice['title']}** on {choice['platform']}")
else:
    print("\n😕 Sorry, no exact matches found. Try changing your inputs!")

