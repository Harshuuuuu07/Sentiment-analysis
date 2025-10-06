import streamlit as st
import pandas as pd
import re
import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('stopwords', quiet=True)

# Updated page config for the movie theme
st.set_page_config(page_title="Movie Review Analyzer", page_icon="🎬", layout="wide")

stemmer = PorterStemmer()

def preprocess_review(review):
    """Clean and preprocess text"""
    # Added a step to remove HTML tags, common in the IMDB dataset
    review = re.sub('<.*?>', ' ', review) 
    review = re.sub('[^a-zA-Z]', ' ', review).lower()
    words = [stemmer.stem(w) for w in review.split() 
             if w not in stopwords.words('english')]
    return ' '.join(words)

@st.cache_resource
def load_model():
    """Load trained model and vectorizer"""
    try:
        model = pickle.load(open('sentiment_model.pkl', 'rb'))
        vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
        return model, vectorizer
    except FileNotFoundError:
        return None, None

# Load model
model, vectorizer = load_model()

# Header
st.title("🎬 Movie Review Sentiment Analyzer")
st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose:", ["Model Info", "Single Review", "Batch Prediction"])

# Model Info
if option == "Model Info":
    st.header("Model Information")
    if model:
        st.success("✅ Model loaded successfully!")
        # Updated model details to reflect the new training
        st.write("**Algorithm:** Multinomial Naive Bayes")
        st.write("**Features:** Count Vectorizer (5000 features)")
        
        features = vectorizer.get_feature_names_out()
        st.write(f"**Vocabulary:** {len(features)} words")
        st.write("Sample features:", ", ".join(features[:20]))
    else:
        st.error("❌ Model not found! Run the updated `train_model.py` first.")

# Single Review Prediction
elif option == "Single Review":
    st.header("Predict a Single Movie Review")
    
    if not model:
        st.warning("⚠️ Train the model first: `python train_model.py`")
    else:
        col1, col2 = st.columns(2)
        with col1:
            # Updated sample review
            if st.button("Positive Sample"):
                st.session_state.sample = "An absolute masterpiece. The acting was incredible and the storyline was captivating from start to finish."
        with col2:
            # Updated sample review
            if st.button("Negative Sample"):
                st.session_state.sample = "A complete waste of time. The plot was predictable and the acting was wooden. I wanted to walk out."
        
        review_text = st.text_area("Enter movie review:", 
                                   value=st.session_state.get('sample', ''),
                                   height=100)
        
        if st.button("Analyze") and review_text:
            processed = preprocess_review(review_text)
            vectorized = vectorizer.transform([processed]).toarray()
            
            prediction = model.predict(vectorized)[0]
            proba = model.predict_proba(vectorized)[0]
            
            col1, col2 = st.columns(2)
            with col1:
                if prediction == 1:
                    st.success("✅ Positive Review")
                    st.balloons()
                else:
                    st.error("❌ Negative Review")
            
            with col2:
                st.metric("Confidence", f"{max(proba) * 100:.1f}%")
            
            # Probability chart
            prob_df = pd.DataFrame({
                'Sentiment': ['Negative', 'Positive'],
                'Probability': proba * 100
            })
            st.bar_chart(prob_df.set_index('Sentiment'))
            
            with st.expander("View processed text"):
                st.write(processed)

# Batch Prediction
elif option == "Batch Prediction":
    st.header("Batch Prediction for Multiple Reviews")
    
    if not model:
        st.warning("⚠️ Train the model first: `python train_model.py`")
    else:
        # Updated uploader text
        uploaded = st.file_uploader("Upload CSV (with 'review' column) or a TXT file", 
                                   type=['csv', 'txt'])
        
        if uploaded:
            try:
                # Load data
                if uploaded.name.endswith('.csv'):
                    df = pd.read_csv(uploaded)
                    # Changed to look for lowercase 'review' column
                    if 'review' not in df.columns:
                        st.error("CSV file must contain a 'review' column.")
                        st.stop()
                else:
                    reviews = uploaded.read().decode('utf-8').splitlines()
                    # Renamed column to 'review' for consistency
                    df = pd.DataFrame(reviews, columns=['review'])
                
                st.subheader("Preview of Uploaded Reviews")
                st.dataframe(df.head())
                
                if st.button("Predict All"):
                    with st.spinner("Processing..."):
                        # Process and predict using the 'review' column
                        processed = df['review'].apply(preprocess_review)
                        vectorized = vectorizer.transform(processed).toarray()
                        predictions = model.predict(vectorized)
                        probas = model.predict_proba(vectorized)
                        
                        # Add results
                        df['Sentiment'] = pd.Series(predictions).map({0: 'Negative', 1: 'Positive'})
                        df['Confidence'] = np.max(probas, axis=1) * 100
                        
                        st.subheader("Prediction Results")
                        st.dataframe(df)
                        
                        # Stats
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Reviews", len(predictions))
                        col2.metric("Positive Reviews", (predictions == 1).sum())
                        col3.metric("Negative Reviews", (predictions == 0).sum())
                        
                        # Download
                        st.download_button(
                            "Download Results as CSV",
                            df.to_csv(index=False),
                            "movie_sentiment_predictions.csv",
                            "text/csv"
                        )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")