import streamlit as st
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import os
from preprocessing import preprocess_text

@st.cache_resource
def load_model(model_path="models/lda_model.gensim", vocab_path="models/vocabulary.dict"):
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
        - **Source**: [BBC Dataset](http://mlg.ucd.ie/datasets/bbc.html)
        - **Documents**: 2,127 unique articles
        - **Algorithm**: Gensim LDA
        - **Representation**: Bag of Words
        """)
        
        st.markdown("## 🔗 Links")
        st.markdown("[📂 GitHub Repository](https://github.com/ahmedxnov/topic-modeling-bbc)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🔍 Article Classification")
        st.markdown("Enter a news article below to predict its topic:")
        
        topic_labels = {0: "Politics", 1: "Business", 2: "Entertainment", 3: "Technology", 4: "Sports"}
        topic_icons = {0: "🏛️", 1: "💼", 2: "🎬", 3: "💻", 4: "⚽"}
        
        article = st.text_area("Enter an article for topic prediction:", height=150, placeholder="Type or paste your news article here...")
        
        predict_button = st.button("🎯 Predict Topic Distribution", type="primary")
    
    with col2:
        st.markdown("## 📋 Model Topics")
        st.markdown("**The model can identify these topics:**")
        for topic_id, topic_name in topic_labels.items():
            st.markdown(f"{topic_icons[topic_id]} **{topic_name}**")
    
    if predict_button:
        if article.strip():
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
        else:
            st.error("Please enter some text for prediction.")

if __name__ == "__main__":
    main()