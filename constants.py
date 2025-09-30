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
AUXILIARY_VERBS = {"could", "would", "shall", "may", "might", "must", "need", "dare", "used", "ought"}
FREQ_ADVERBS = {"always", "usually", "sometimes", "often", "rarely", "seldom", "never", "ever", "hardly", "occasionally", "frequently", "generally", "normally", "regularly"}
GENERIC_ADVERBS = {"maybe", "really", "already", "still", "yet", "also", "even", "perhaps", "quite", "almost", "us"}
GENERIC_VERBS = {"make", "made", "take", "took", "put", "keep", "let", "say", "said", "tell", "told", "think", "thought", "know", "knew", "want", "wanted", "like", "liked", "use", "used", "give", "gave", "get", "got", "find", "found", "see", "saw", "go", "went", "come", "came", "leave", "left", "try", "tried", "work", "worked", "call", "called", "look", "looked"}
GENERIC_INTENSIFIERS = {"big", "small", "great", "nice", "bad", "good", "new", "old", "high", "low", "many", "much", "less", "least", "better", "best", "worse", "worst", "several", "various", "another", "different", "enough", "plenty", "every"}

NUMERICS = {"first","second","third","fourth","fifth","sixth","seventh","eighth","ninth","tenth","one","two","three","four","five","six","seven","eight","nine","ten"}
TRANSITION_WORDS = {"however","nevertheless","nonetheless","although","though","whereas","instead","yet","despite","still","therefore","thus","hence","consequently","accordingly","since","moreover","furthermore", "also","besides","additionally","next","afterward","afterwards","subsequently","finally","earlier","later","meantime","meanwhile","indeed","namely","specifically","overall","ultimately"}
TIME_WORDS = {"morning","afternoon","evening","night","midnight","dawn","dusk","day","week","month","year","weekend","weekday","spring","summer","autumn","fall","winter","yesterday","today","tomorrow","tonight","soon","later","earlier","presently","eventually","immediately","forthwith","shortly","afterward","afterwards","beforehand","meantime","meanwhile"}
HONORIFICS = {"mr","mrs","ms","mx","dr","prof","sir"}
REPORTING_VERBS = {
    "claim", "claims", "claimed", "claiming",
    "state", "states", "stated", "stating",
    "argue", "argues", "argued", "arguing",
    "add", "adds", "added", "adding",
    "note", "notes", "noted", "noting",
    "remark", "remarks", "remarked", "remarking",
    "comment", "comments", "commented", "commenting",
    "report", "reports", "reported", "reporting",
    "announce", "announces", "announced", "announcing",
    "confirm", "confirms", "confirmed", "confirming",
    "deny", "denies", "denied", "denying",
    "admit", "admits", "admitted", "admitting",
    "allege", "alleges", "alleged", "alleging",
    "explain", "explains", "explained", "explaining",
    "assert", "asserts", "asserted", "asserting",
    "maintain", "maintains", "maintained", "maintaining",
    "contend", "contends", "contended", "contending",
    "suggest", "suggests", "suggested", "suggesting",
    "estimate", "estimates", "estimated", "estimating",
    "predict", "predicts", "predicted", "predicting",
    "believe", "believes", "believed", "believing",
    "expect", "expects", "expected", "expecting",
    "warn", "warns", "warned", "warning",
    "caution", "cautions", "cautioned", "cautioning",
    "urge", "urges", "urged", "urging",
    "request", "requests", "requested", "requesting",
    "stress", "stresses", "stressed", "stressing",
    "emphasize", "emphasizes", "emphasized", "emphasizing",
    "highlight", "highlights", "highlighted", "highlighting",
    "underline", "underlines", "underlined", "underlining",
    "acknowledge", "acknowledges", "acknowledged", "acknowledging",
    "concede", "concedes", "conceded", "conceding",
    "insist", "insists", "insisted", "insisting"
}

STOPWORDS = BASE | AUXILIARY_VERBS | FREQ_ADVERBS | GENERIC_ADVERBS | GENERIC_VERBS | GENERIC_INTENSIFIERS | NUMERICS | TRANSITION_WORDS | TIME_WORDS | HONORIFICS | REPORTING_VERBS