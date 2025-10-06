# train_model.py
import pandas as pd
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
# Switched to MultinomialNB, which is better suited for text classification
from sklearn.naive_bayes import MultinomialNB 
import nltk

# Download NLTK data quietly
nltk.download('stopwords', quiet=True)

stemmer = PorterStemmer()

def preprocess_review(review_text):
    """Clean and preprocess a single review."""
    # Remove HTML tags and non-alphabetic characters
    review_text = re.sub('<.*?>', ' ', review_text) # Remove HTML tags
    review_text = re.sub('[^a-zA-Z]', ' ', review_text).lower()
    
    # Tokenize, stem, and remove stopwords
    words = [stemmer.stem(w) for w in review_text.split() 
             if w not in stopwords.words('english')]
    return ' '.join(words)

# --- Main script execution ---

# 1. Load the IMDB dataset
print("Loading IMDB dataset...")
# Ensure your CSV file is named 'IMDB Dataset.csv' or change the filename below
df = pd.read_csv('IMDB_Dataset.csv')
print(f"Dataset loaded with {len(df)} reviews.")

# 2. Map sentiment labels to numbers (0 for negative, 1 for positive)
print("Mapping sentiment labels to numbers...")
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# 3. Preprocess the review text
print("Preprocessing all reviews... (this may take a few minutes)")
# Apply the preprocessing function to the 'review' column
corpus = df['review'].apply(preprocess_review).tolist()
y = df['sentiment'].tolist()

# 4. Vectorize text data into features
print("Vectorizing text data...")
# Increased max_features for the larger dataset
cv = CountVectorizer(max_features=5000) 
X = cv.fit_transform(corpus).toarray()

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 6. Train the Naive Bayes model
print("Training the Multinomial Naive Bayes model...")
# Using MultinomialNB which is ideal for word counts
clf = MultinomialNB()
clf.fit(X_train, y_train)

# 7. Evaluate the model
accuracy = clf.score(X_test, y_test) * 100
print(f"Model Accuracy: {accuracy:.2f}%")

# 8. Save the trained model and the vectorizer
print("Saving model and vectorizer to disk...")
pickle.dump(clf, open('sentiment_model.pkl', 'wb'))
pickle.dump(cv, open('vectorizer.pkl', 'wb'))

print("Model training complete and files saved successfully!")