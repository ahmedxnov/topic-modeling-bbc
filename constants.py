import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

TOKENIZER = word_tokenize
LEMMATIZER = WordNetLemmatizer()
PATTERNS = [
    # General cleaning patterns
    ("ctrl_ws", re.compile(r"[\r\n\t]+"), " "),
    ("html_tags", re.compile(r"<[^>]+>"), " "),
    ("html_entities", re.compile(r"&[A-Za-z0-9#]+;"), " "),
    ("urls", re.compile(r"https?://\S+|www\.\S+"), " "),
    ("emails", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), " "),
    ("handles", re.compile(r"(?<!\w)@[A-Za-z0-9_]+"), " "),
    ("ampersand_to_and", re.compile(r"&"), " and "),
    ("pounds_symbol", re.compile(r"£"), " pounds "),
    ("dollars_symbol", re.compile(r"\$"), " dollars "),
    ("numbers", re.compile(r"\b\w*\d+\w*\b"), " "),
    ("nonword_symbols", re.compile(r"[^A-Za-z\s]"), " "),
    ("multi_space", re.compile(r"\s{2,}"), " ")
]


BASE = set(stopwords.words("english"))
CONTRACTED_AUXILIARY_ENDINGS = {'m', "ve", 'l', 'd', 're', 's'}
AUXILIARY_VERBS = {"am", "is", "are", "was", "were", "be", "being", "been", "have", "has", "had", "do", "does", "did", "can", "could", "will", "would", "shall", "should", "may", "might", "must", "need", "dare", "used", "ought"}
FREQ_ADVERBS = {"always", "usually", "sometimes", "often", "rarely", "seldom", "never", "ever", "hardly", "occasionally", "frequently", "generally", "normally", "regularly"}
GENERIC_ADVERBS = {"maybe", "really", "just", "already", "again", "still", "yet", "also", "even", "perhaps", "quite", "so", "too", "very", "almost"}
GENERIC_VERBS = {"make", "made", "take", "took", "put", "keep", "let", "say", "said", "tell", "told", "think", "thought", "know", "knew", "want", "wanted", "like", "liked", "use", "used", "give", "gave", "get", "got", "find", "found", "see", "saw", "go", "went", "come", "came", "leave", "left", "try", "tried", "work", "worked", "call", "called", "look", "looked"}
GENERIC_INTENSIFIERS = {"big", "small", "great", "nice", "bad", "good", "new", "old", "high", "low", "many", "much", "more", "most", "less", "least", "better", "best", "worse", "worst", "few", "several", "various", "other", "another", "same", "different", "enough", "plenty", "some", "any", "all", "every", "each"}
STOPWORDS = BASE | CONTRACTED_AUXILIARY_ENDINGS | AUXILIARY_VERBS | FREQ_ADVERBS | GENERIC_ADVERBS | GENERIC_VERBS | GENERIC_INTENSIFIERS | {"us"}