import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import os

# --- 1. Data Loading and Preprocessing (run once) ---
file_path = ("C://Project2//IMDB Dataset.csv")

# Load the full dataset to train the model
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    exit()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['review'] = df['review'].apply(clean_text)

# --- 2. Train the Model (run once) ---
# Feature Engineering - Train the vectorizer on the entire dataset
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(df['review'])
y = df['sentiment']

# Model Training
model = MultinomialNB()
model.fit(X, y)

# --- 3. Flask Web Application ---
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    if request.method == 'POST':
        # Get the text from the web form
        user_text = request.form['user_input']
        
        # Clean the text using the same function
        cleaned_text = clean_text(user_text)
        
        # Transform the text using the trained TF-IDF vectorizer
        transformed_text = tfidf_vectorizer.transform([cleaned_text])
        
        # Make a prediction
        prediction = model.predict(transformed_text)[0]
        
        # Display the result
        prediction_text = f"The sentiment of the text is: {prediction.upper()}"
        
    return render_template('sentiment_ui.html', prediction=prediction_text)

if __name__ == '__main__':
    # Using a different port (e.g., 5001) in case port 5000 is still in use
    app.run(debug=True, port=5001)