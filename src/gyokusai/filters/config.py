from pathlib import Path

BAD_WORDS_PATH = Path(__file__).parents[1] / "word_lists" / "badwords_en.txt"
ENGLISH_STOPWORDS = ["the", "be", "to", "of", "and", "that", "have", "with"]