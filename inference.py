import argparse
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import os
from preprocessing import preprocess_text
def main():
    parser = argparse.ArgumentParser(description="Topic Modeling Inference")
    parser.add_argument("--model_path", type=str, default="models/lda_model.gensim", help="Path to the trained LDA model")
    parser.add_argument("--vocab_path", type=str, default="models/vocabulary.dict", help="Path to the vocabulary dictionary")
    parser.add_argument("--new_data", type=str, required=True, help="Path to the new dataset text file for inference")
    args = parser.parse_args()
    

    if not args.new_data:
        raise ValueError("Please provide path for new data using --new_data argument.")
    
    
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model file not found at {args.model_path}")
    if not os.path.isfile(args.vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found at {args.vocab_path}")
    
    print(f"Loading model from {args.model_path}")
    print(f"Loading vocabulary from {args.vocab_path}")
    
    try:
        lda_model = LdaModel.load(args.model_path)
        vocabulary = Dictionary.load(args.vocab_path)
        print("Model and vocabulary loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Error loading model or vocabulary: {e}")
    
    
    if not os.path.isfile(args.new_data):
        raise FileNotFoundError(f"New data file not found at {args.new_data}")
    
    try:
        with open(args.new_data, 'r', encoding='utf-8') as file:
            document_text = file.read().strip()
        
        if not document_text:
            raise ValueError("The input document is empty")
            
        print(f"Loaded new document for inference from {args.new_data}")
    except Exception as e:
        raise RuntimeError(f"Error reading new data file: {e}")

    try:
        preprocessed_doc = preprocess_text(document_text)
        if not preprocessed_doc:
            raise ValueError("Document became empty after preprocessing")
        print("Document preprocessing completed")
    except Exception as e:
        raise RuntimeError(f"Error during preprocessing: {e}")
    
    try:
        BoW = vocabulary.doc2bow(preprocessed_doc)
        if not BoW:
            raise ValueError("No words in document found in vocabulary")
        
        topics = lda_model.get_document_topics(BoW, minimum_probability=0.0)
        
        print("\nTopic predictions:")
        for topic_id, probability in topics:
            print(f"Topic {topic_id}: {probability:.4f}")
            
    except Exception as e:
        raise RuntimeError(f"Error during inference: {e}")


if __name__ == "__main__":
    main()