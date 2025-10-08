#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import os
from src.utils.preprocessing import preprocess_text
from src.utils.constants import MODEL_PATH, VOCAB_PATH

@st.cache_resource
def load_model(model_path=MODEL_PATH, vocab_path=VOCAB_PATH):
    model_path = str(model_path)
    vocab_path = str(vocab_path)
    
    if not os.path.isfile(model_path):
        st.error(f"Model file not found at {model_path}")
        return None, None
    if not os.path.isfile(vocab_path):
        st.error(f"Vocabulary file not found at {vocab_path}")
        return None, None
    
    try:
        lda_model = LdaModel.load(model_path)
        vocabulary = Dictionary.load(vocab_path)
        return lda_model, vocabulary
    except Exception as e:
        st.error(f"Error loading model or vocabulary: {e}")
        return None, None


def main():
    try:
        st.set_page_config(
            page_title="BBC News Topic Classifier", 
            page_icon="📰",
            layout="wide"
        )
    except:
        pass  
    
    lda_model, vocabulary = load_model()
    
    if lda_model is None or vocabulary is None:
        st.stop()
    
    st.markdown("# 📰 BBC News Topic Classifier")
    st.markdown("**Powered by LDA Topic Modeling**")
    
    with st.sidebar:
        st.markdown("## ℹ️ About")
        st.markdown("""
        This model classifies BBC news articles into **5 topics**:
        - 🏛️ **Politics** 
        - 💼 **Business**
        - 🎬 **Entertainment**
        - 💻 **Technology** 
        - ⚽ **Sports**
        """)
        
        st.markdown("## 📊 Dataset Info")
        st.markdown("""
        - **Name**: BBC News Articles Datase
        - **Documents**: 2,127 unique articles
        - **Algorithm**: Gensim LDA
        - **Representation**: Bag of Words
        """)
        
        st.markdown("## 🔗 Links")
        st.markdown("📈 [LDA Paper](https://arxiv.org/pdf/1711.04305)")
        st.markdown("🌐 [BBC Dataset](http://mlg.ucd.ie/datasets/bbc.html)")
        st.markdown("[📂 GitHub Repository](https://github.com/ahmedxnov/topic-modeling-bbc)")
        st.markdown("---")
        st.markdown("**👨‍💻 Developer:** [Ahmad Khaled](https://www.linkedin.com/in/ahmad-khaled-hamed/)")
    
    # 🔹 Everything now in one column
    st.markdown("## 🔍 Article Classification")
    st.markdown("Enter a news article below to predict its topic:")
    
    topic_labels = {0: "Politics", 1: "Business", 2: "Entertainment", 3: "Technology", 4: "Sports"}
    topic_icons = {0: "🏛️", 1: "💼", 2: "🎬", 3: "💻", 4: "⚽"}
    
    article = st.text_area("Enter an article for topic prediction:", height=150, placeholder="Type or paste your news article here...")
    
    predict_button = st.button("🎯 Predict Topic Distribution", type="primary")
    
    # Handle prediction logic
    if predict_button and article and article.strip():
        try:
            processed_article = preprocess_text(article)
            
            if not processed_article:
                st.warning("The text became empty after preprocessing. Please try with different text.")
                return
            
            BoW = vocabulary.doc2bow(processed_article)
            
            if not BoW:
                st.warning("No words in the text were found in the model's vocabulary. Please try with different text.")
                return
            
            topic_predictions = lda_model.get_document_topics(BoW, minimum_probability=0.0)
            
            st.divider()
            st.markdown("## 🎯 Prediction Results")
            
            max_topic_id, max_probability = max(topic_predictions, key=lambda x: x[1])
            max_topic_name = topic_labels[max_topic_id]
            max_topic_icon = topic_icons[max_topic_id]
            
            st.success(f"{max_topic_icon} **Most likely topic: {max_topic_name}** ({max_probability:.1%} confidence)")
            
            st.markdown("**All Topic Probabilities:**")
            
            cols = st.columns(len(topic_labels))
            for i, (topic_id, probability) in enumerate(topic_predictions):
                topic_name = topic_labels[topic_id]
                topic_icon = topic_icons[topic_id]
                with cols[i]:
                    st.metric(
                        label=f"{topic_icon} {topic_name}",
                        value=f"{probability:.1%}"
                    )
                    
            with st.expander("📋 Model Information & Limitations"):
                st.info("""
                **Model Training:** This LDA model was trained specifically on BBC news articles.
                
                **Best Performance:** Input articles similar in style and content to BBC news for optimal results.
                
                **Limitations:**
                - May not generalize well to non-news text or informal writing
                - Doesn't handle grammatical errors or heavy slang
                - Probabilities may not sum to exactly 100% due to rounding
                """)
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    elif predict_button:
        st.error("Please enter some text for prediction.")

if __name__ == "__main__":
    main()