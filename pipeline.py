from data import *
import argparse
from vectorizer import *
from model import build_lda_model
from gensim.models.phrases import Phrases, Phraser
import yaml


def main():
    parser = argparse.ArgumentParser(description="Topic Modeling Pipeline")
    parser.add_argument("--dataset", type=str, default="dataset/Labeled BBC.csv", help="Path to the dataset CSV file")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file")
    args = parser.parse_args()
    
    
    dataset = read_dataset(args.dataset)
    c_count, chunk_size = cpu_info(len(dataset))
    tokenized_docs = parallel_tokenization(dataset, c_count, chunk_size)
    
    bigram = Phrases(tokenized_docs, min_count=5, threshold=100)
    bigram_mod = Phraser(bigram)
    phrased_docs = [bigram_mod[doc] for doc in tokenized_docs]
    
    preprocessed_documents = parallel_stopword_removal_lemmatize(phrased_docs, c_count, chunk_size)
    
    
    
    vocabulary = build_vocabulary(preprocessed_documents)
    BoW_corpus = build_BoW_corpus(preprocessed_documents, vocabulary)
    
    try:
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error loading config file: {e}") from e
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing config file: {e}") from e


    lda_model = build_lda_model(BoW_corpus, vocabulary, **config)    
    topics = lda_model.print_topics(num_words=10)
    
    for topic_id, topic_str in topics:
        words = [w.split('*')[1].strip().strip('"') for w in topic_str.split('+')]
        print(f"{topic_id}: {', '.join(words)}")

        
    


if __name__ == "__main__":
    main()

