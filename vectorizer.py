from gensim import corpora

def build_vocabulary(documents: list[list[str]]) -> corpora.Dictionary:
    return corpora.Dictionary(documents)
    
    
def build_BoW_corpus(documents: list[list[str]], vocabulary: corpora.Dictionary) -> list[list[tuple[int, int]]]:
    return [vocabulary.doc2bow(doc) for doc in documents]