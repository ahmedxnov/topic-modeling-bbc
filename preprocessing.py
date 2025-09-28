from constants import TOKENIZER, STOPWORDS, LEMMATIZER, PATTERNS, BASE

def clean_text(text : str) -> str:
    text = text.lower()
    
    if PATTERNS:
        for name, pattern, repl in PATTERNS:
            text = pattern.sub(repl, text)
        return text.strip()
    else:
        print("Patterns were empty which is used to clean the text.")
        return text.strip()
  

def tokenize_text(text : str) -> list[str]:
    try:
        return TOKENIZER(text)
    except Exception as e:
        print(f"The error is: {e}")
        return list()

def remove_stopwords(tokens : list[str]) -> list[str]:
    filtered_tokens = list()
    for token in tokens:
        if token not in STOPWORDS:
            filtered_tokens.append(token)
    if filtered_tokens:
        return filtered_tokens
    else:
        print("All tokens were stopwords.")
        return filtered_tokens


def lemmatize_tokens(tokens : list[str]) -> list[str]:
    lemmatized_tokens = list()
    for token in tokens:
        lemmatized_tokens.append(LEMMATIZER.lemmatize(token))
    return lemmatized_tokens
        
        
def preprocess_text(text: str) -> list[str]:
    assert isinstance(text, str), "Input must be a string."
    if text.strip():
        return lemmatize_tokens(remove_stopwords(tokenize_text(clean_text(text))))
    else:
        print("The provided text is empty or whitespace.")
        return list()



