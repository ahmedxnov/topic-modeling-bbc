from constants import *
from nltk import pos_tag
import pandas as pd
def clean_and_tokenize(text : str) -> list[str]:
    text = text.strip()
    for _, pattern, repl in PATTERNS:
        text = pattern.sub(repl, text) 
    return TOKENIZER(text)

def filter_nouns_and_lemmatize(tokens: list[str]) -> list[str]:
    filtered_tokens = list()
    pos_tags = pos_tag(tokens)
    for word, tag in pos_tags:
        if tag == "NN" or tag == "NNS":
            filtered_tokens.append(LEMMATIZER.lemmatize(word, pos='n'))
    return filtered_tokens

def preprocess_text(text: str) -> list[str]:
    return filter_nouns_and_lemmatize(clean_and_tokenize(text))