from gensim.models import LdaModel

def build_lda_model(corpus,vocabulary,**params) -> LdaModel:
    Lda = LdaModel(corpus=corpus, id2word=vocabulary, **params)
    return Lda