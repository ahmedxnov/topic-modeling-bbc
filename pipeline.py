from data import *
import argparse
from vectorizer import *
from model import build_lda_model
import yaml
import sys


def main():
    parser = argparse.ArgumentParser(description="Topic Modeling Pipeline")
    parser.add_argument("--dataset", type=str, default="dataset/Labeled BBC.csv", help="Path to the dataset CSV file")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file")
    args = parser.parse_args()
    
    
    dataset = read_dataset(args.dataset)
    c_count, chunk_size = cpu_info(len(dataset))
    preprocessed_documents = parallel_process(dataset, c_count, chunk_size)
    
    vocabulary = build_vocabulary(preprocessed_documents)
    BoW_corpus = build_BoW_corpus(preprocessed_documents, vocabulary)
    
    try:
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError as e:
        sys.exit(f"Error loading config file: {e}")
    except yaml.YAMLError as e:
        sys.exit(f"Error parsing config file: {e}")
    except Exception as e:
        sys.exit(f"Unexpected error: {e}")

    lda_model = build_lda_model(BoW_corpus, vocabulary, **config)
    
    topics = lda_model.print_topics(num_words=10)
    for topic in topics:  
        print(type(topic))


if __name__ == "__main__":
    main()

