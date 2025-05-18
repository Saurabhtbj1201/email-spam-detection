from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import logging
import os

# Set up logging
logging.basicConfig(level=logging.WARNING)  # Reduce log level to WARNING
logger = logging.getLogger(__name__)

# Download NLTK data with error handling
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        logger.info("NLTK data downloaded successfully")
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {str(e)}")
        raise

download_nltk_data()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load and preprocess dataset
try:
    data = pd.read_csv('data/spam.csv', encoding='latin-1')
    data = data[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
    data['label'] = data['label'].map({'safe': 0, 'spam': 1})
    data = data.dropna(subset=['text', 'label'])
    # logger.info(f"Dataset loaded with {len(data)} entries")
except Exception as e:
    logger.error(f"Failed to load dataset: {str(e)}")
    raise

# Check dataset size
if len(data) < 2:
    logger.error("Not enough data to split into training and testing sets")
    raise ValueError("Dataset too small")

X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Extend stopwords
custom_stopwords = set(stopwords.words('english')).union({'subject', 'http', 'www', 'com'})

# Preprocess message with error handling
def preprocess_message(message):
    try:
        message = message.lower().strip()
        message = re.sub(r'http\S+|www\S+|@\S+|\d+', '', message)  # Remove URLs, mentions, numbers
        message = re.sub(r'[^\w\s]', '', message)  # Remove punctuation
        words = message.split()
        words = [word for word in words if word not in custom_stopwords]  # Remove stopwords
        lemmatizer = WordNetLemmatizer()
        processed_words = []
        for word in words:
            try:
                lemma = lemmatizer.lemmatize(word)
                processed_words.append(lemma)
            except Exception as e:
                logger.warning(f"Failed to lemmatize word '{word}': {str(e)}")
                processed_words.append(word)  # Fallback to original word
        return ' '.join(processed_words)
    except Exception as e:
        logger.error(f"Error preprocessing message: {str(e)}")
        return message  # Fallback to original message

# Use a smaller vectorizer to save memory
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=2000)

# Apply vectorization
try:
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    # logger.info("Vectorization completed")
except Exception as e:
    logger.error(f"Vectorization failed: {str(e)}")
    raise

# Remove SMOTE to save memory
# Compute class weights
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

# Train a smaller model without GridSearchCV
try:
    model = RandomForestClassifier(
        random_state=42,
        class_weight=class_weights_dict,
        n_estimators=30,      # Fewer trees
        max_depth=8,          # Shallower trees
        min_samples_split=5   # Slightly higher min samples
    )
    model.fit(X_train_vec, y_train)
    # logger.info("Model training completed")
except Exception as e:
    logger.error(f"Model training failed: {str(e)}")
    raise

# Remove evaluation metrics to save memory

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_message = request.json.get('message', '')
        # logger.debug(f"Received message: {user_message}")
        if not user_message:
            # logger.warning("No message provided in request")
            return jsonify({'error': 'No message provided'}), 400

        # Preprocess the input message
        user_message_processed = preprocess_message(user_message)
        # logger.debug(f"Preprocessed message: {user_message_processed}")

        # Transform and predict
        user_message_vec = vectorizer.transform([user_message_processed])
        prediction = model.predict(user_message_vec)[0]
        # logger.debug(f"Prediction result: {prediction}")

        result = 'spam' if prediction == 1 else 'safe'
        return jsonify({'message': user_message, 'prediction': result})
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Add a root route for health/status
@app.route('/')
def index():
    return "Email Spam Detection API is running."

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)