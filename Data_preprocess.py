import nltk
import re
from collections import defaultdict
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode
from collections import Counter

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Initialize tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Additional custom stopwords (optional)
custom_stopwords = {
    "will", "new", "york", "years", "one", "said", "united", "state", "trump", "time",
    "country", "two", "city", "china", "first", "woman", "american", "make", "made",
    "work", "company", "take", "family", "president", "government", "plan", "life",
    "people", "say", "says", "saying", "may", "show", "look", "help", "many", "home",
    "year", "day", "even", "women", "team", "teams", "states", "child", "russia", "would",
    "part", "world", "want", "set", "way", "found", "group", "played", "playing", "time", "election", "charge",
    "player", "play", "countries", "country", "plays", "become", "becomes", "became", "right",
    "three", "come", "needing", "came", "comes", "weeks", "week", "need", "needed",
    "needs", "official", "still", "including", "former", "last", "party", "star",
    "back", "place", "change", "return", "leader", "offer", "history", "season",
    "support", "couple", "met", "know", "find", "hope", "others", "power", "game",
    "talk", "toke", "token", "call", "called", "calling", "calls", "million", "can",
    "give", "given", "giving", "gives", "go", "going", "gone", "goes", "could", "get", "also", "open",
    "take", "taken", "taking", "takes", "bring", "bringing", "brought", "brings",
    "old", "run", "running", "ran", "runs", "use", "used", "using", "uses",
    "try", "trying", "tried", "tries", "artist", "business", "police", "report", "protest", "case", "start", "started",
    "starts", "end", "ending", "ended", "much", "big", "large", "top", "official", "case", "month", "plan", "appear", "live", "long", "man",
    "move", "moves", "moved", "moving", "tell", "tells", "told", "telling", "face", "faces", "faced", "facing",
    "show", "shows", "showed", "showing", "know", "knows", "knew", "knowing", "offer", "offers", "offered", "offering",
    "begin", "begins", "began", "beginning", "hold", "holds", "held", "holding", "put", "puts", "putting", "took",
    "bring", "brings", "brought", "bringing", "call", "calls", "called", "calling", "run", "runs", "ran", "running", "use", "uses", "used", "using",
    "try", "tries", "tried", "trying", "see", "sees", "saw", "seeing", "seen", "seening"
}
all_stopwords = stop_words.union(custom_stopwords)

# Function to normalize text (lowercase, remove special characters)
def normalize_text(text):
    """
    Normalize the text by making it lowercase, removing accents, 
    and removing non-alphabetic characters.
    """
    text = text.lower().strip()
    text = unidecode(text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Function to map POS tags for better lemmatization
def get_wordnet_pos_bulk(words):
    """
    Map POS tags to WordNet's POS tags for better lemmatization.
    """
    tag_dict = defaultdict(lambda: wordnet.NOUN, {"J": wordnet.ADJ, "V": wordnet.VERB, "R": wordnet.ADV})
    return [(word, tag_dict.get(pos[0].upper(), wordnet.NOUN)) for word, pos in nltk.pos_tag(words)]

# Main preprocessing function
def preprocess_text(text):
    """
    Preprocess text by normalizing, removing stopwords, tokenizing,
    POS tagging, and lemmatizing the words.
    """
    if not isinstance(text, str) or len(text) < 10:
        return None

    # Normalize text
    text = normalize_text(text)

    # Tokenize and remove stopwords
    words = [word for word in word_tokenize(text) if word not in all_stopwords and len(word) > 2]

    # POS tagging and lemmatization
    tagged_words = get_wordnet_pos_bulk(words)
    cleaned_words = [lemmatizer.lemmatize(word, pos) for word, pos in tagged_words]

    return ' '.join(cleaned_words) if cleaned_words else None
