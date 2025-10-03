import re
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
    ("currencie_to_dollar", re.compile(r"[£€¥₹₽₩₺₦₫₪₱฿₴₡₲₵₸₭₨]"), "$"),
    ("remove_percent", re.compile(r"%"), ""),
    ("multi_space", re.compile(r"\s{2,}"), " ")
]