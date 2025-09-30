from constants import TOKENIZER, STOPWORDS, LEMMATIZER, PATTERNS

def clean_text(text : str) -> str:
    text = text.lower()
    for name, pattern, repl in PATTERNS:
        text = pattern.sub(repl, text) 
    return text.strip()
  

def tokenize_text(text : str) -> list[str]:
    return TOKENIZER(text)

def remove_stopwords(tokens : list[str]) -> list[str]:
    filtered_tokens = list()
    for token in tokens:
        if token not in STOPWORDS:
            filtered_tokens.append(token)
    return filtered_tokens


def lemmatize_tokens(tokens : list[str]) -> list[str]:
    lemmatized_tokens = list()
    for token in tokens:
        lemmatized_tokens.append(LEMMATIZER.lemmatize(token))
    return lemmatized_tokens
        
   
def clean_and_tokenize(text: str) -> list[str]:
    return tokenize_text(clean_text(text))   

def remove_stopwords_lemmatize(tokens : list[str]) -> list[str]:
    return lemmatize_tokens(remove_stopwords(tokens))