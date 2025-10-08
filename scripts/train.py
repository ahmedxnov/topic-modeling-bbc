import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset_pipeline import *
import argparse
from gensim.models import LdaModel
from gensim import corpora
import yaml
import os
def main():
    parser = argparse.ArgumentParser(description="Topic Modeling Pipeline")
    parser.add_argument("--dataset", type=str, default="dataset/Labeled BBC.csv", help="Path to the dataset CSV file")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the configuration file")
    parser.add_argument("--save_model", type=str, default="models/lda_model.gensim", help="Path to save the trained LDA model")
    parser.add_argument("--save_vocab", type=str, default="models/vocabulary.dict", help="Path to save the vocabulary dictionary")
    args = parser.parse_args()
    
    print("Starting Topic Modeling Pipeline...")
    print(f"Dataset: {args.dataset}")
    print(f"Config: {args.config}")
    print(f"Save Model: {args.save_model}")
    
    print("Step 1: Loading dataset...")
    dataset = read_dataset(args.dataset)
    print(f"Loaded {len(dataset)} documents")
    
    print("\nStep 2: Setting up parallel processing...")
    c_count, chunk_size = cpu_info(len(dataset))
    print(f"Using {c_count} cores with chunk size {chunk_size}")
    
    print("\nStep 3: Preprocessing documents...")
    tokenized_docs = parallel_preprocessing(dataset, c_count, chunk_size)
    print("Document preprocessing completed")
    
    print("\nStep 4: Building vocabulary...")
    vocabulary = corpora.Dictionary(tokenized_docs)
    print(f"Initial vocabulary size: {len(vocabulary)}")
    
    print("Filtering vocabulary extremes...")
    vocabulary.filter_extremes(no_below=30 ,no_above=0.2)
    vocabulary.compactify()
    print(f"Filtered vocabulary size: {len(vocabulary)}")
    
    print("\nStep 5: Creating Bag of Words corpus...")
    BoW_corpus = [vocabulary.doc2bow(doc) for doc in tokenized_docs]
    print("BoW corpus created")
    
    print("\nStep 6: Loading configuration...")
    try:
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
        print(f"Configuration loaded from {args.config}")
        print(f"Model parameters: {config}")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error loading config file: {e}") from e
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing config file: {e}") from e

    print("\nStep 7: Training LDA model...")
    lda_model = LdaModel(corpus=BoW_corpus, id2word=vocabulary, **config)
    print("LDA model training completed")
    
    print("\nStep 8: Extracting topics...")
    topics = lda_model.print_topics(num_words=10)
    print(f"Extracted {len(topics)} topics")
    
    print("\nStep 9: Saving model and vocabulary...")
    os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
    os.makedirs(os.path.dirname(args.save_vocab), exist_ok=True)
    
    lda_model.save(args.save_model)
    print(f"Model saved to {args.save_model}")

    vocabulary.save(args.save_vocab)
    print(f"Vocabulary saved to {args.save_vocab}")

    print("\nStep 10: Displaying discovered topics...")
    for topic_id, topic_str in topics:
        words = [w.split('*')[1].strip().strip('"') for w in topic_str.split('+')]
        print(f"Topic {topic_id}: {', '.join(words)}")
    
    print("\nPipeline completed successfully!")
if __name__ == "__main__":
    main()