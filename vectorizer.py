from gensim import corpora

def build_vocabulary(documents: list[list[str]]) -> corpora.Dictionary:
    if not documents:
        raise ValueError("Documents list cannot be empty")
    try:
        vocabulary = corpora.Dictionary(documents)
        if len(vocabulary) == 0:
            raise ValueError("No valid tokens found in documents")
        return vocabulary
    except Exception as e:
        print(f"Error building vocabulary: {e}")
        raise
    

def build_BoW_corpus(documents: list[list[str]], vocabulary: corpora.Dictionary) -> list[list[tuple[int, int]]]:
    try:
        corpus = [vocabulary.doc2bow(doc) for doc in documents]
        return corpus
    except Exception as e:
        print(f"Error building BoW corpus: {e}")
        return []