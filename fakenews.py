import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import streamlit as st

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Function for stemming
def stemming(content):
    port_stem = PorterStemmer()
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Function to train and save the model (run this once)
def train_and_save_model(data_path):
    # Load the dataset
    news_dataset = pd.read_csv(data_path)
    
    # Fill missing values
    news_dataset = news_dataset.fillna('')
    
    # Combine author and title into content
    news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']
    
    # Apply stemming to content
    news_dataset['content'] = news_dataset['content'].apply(stemming)
    
    # Separate features and target
    X = news_dataset['content'].values
    Y = news_dataset['label'].values
    
    # Initialize and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    vectorizer.fit(X)
    X = vectorizer.transform(X)
    
    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    
    # Train the model
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    
    # Save the model and vectorizer
    with open('fake_news_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Evaluate the model
    X_test_prediction = model.predict(X_test)
    test_accuracy = accuracy_score(X_test_prediction, Y_test)
    
    return test_accuracy

# Function to load the model and vectorizer
def load_model():
    try:
        with open('fake_news_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found. Please train the model first.")
        return None, None

# Function to make predictions
def predict_news(text, author=""):
    model, vectorizer = load_model()
    
    if model is None or vectorizer is None:
        return None
    
    # Combine author and text
    content = author + " " + text
    
    # Preprocess the content
    processed_content = stemming(content)
    
    # Vectorize the content
    vectorized_content = vectorizer.transform([processed_content])
    
    # Make prediction
    prediction = model.predict(vectorized_content)
    
    return prediction[0]

# Streamlit web app
def main():
    st.set_page_config(
        page_title="Fake News Detector",
        page_icon="📰",
        layout="centered"
    )
    
    st.title("📰 Fake News Detector")
    st.markdown("### Check if a news article is real or fake")
    
    # Navigation sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Predict News", "Train Model", "About"])
    
    if page == "Predict News":
        st.subheader("Enter News Details")
        
        author = st.text_input("Author (optional)")
        news_text = st.text_area("News Text", height=200)
        
        if st.button("Predict"):
            if news_text.strip() == "":
                st.warning("Please enter some news text.")
            else:
                with st.spinner("Analyzing..."):
                    result = predict_news(news_text, author)
                    
                if result is not None:
                    st.subheader("Prediction Result")
                    if result == 0:
                        st.success("This news appears to be REAL")
                    else:
                        st.error("This news appears to be FAKE")
    
    elif page == "Train Model":
        st.subheader("Train the Fake News Detection Model")
        st.markdown("Upload your dataset to train the model. The dataset should have columns for 'author', 'title', and 'label'.")
        
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file is not None:
            if st.button("Train Model"):
                with st.spinner("Training model... This may take a few minutes."):
                    # Save the uploaded file
                    with open("train.csv", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Train and save the model
                    accuracy = train_and_save_model("train.csv")
                    
                    st.success(f"Model trained successfully! Test accuracy: {accuracy:.4f}")
    
    elif page == "About":
        st.subheader("About Fake News Detector")
        st.markdown("""
        This application uses machine learning to predict whether a news article is real or fake based on its content.
        
        ### How it works:
        1. The app uses a Logistic Regression model trained on a dataset of labeled news articles
        2. Text preprocessing includes stemming and TF-IDF vectorization
        3. The model analyzes patterns in language and content to classify news as real or fake
        
        ### Limitations:
        - The model is only as good as the data it was trained on
        - It may not catch sophisticated fake news that mimics real news writing styles
        - Context and additional research are always recommended when evaluating news sources
        
        ### Privacy:
        - Your inputs are processed locally and not stored
        """)

if __name__ == "__main__":
    main()