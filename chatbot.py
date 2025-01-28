import os
from flask import Flask, render_template, request, jsonify, session
import speech_recognition as sr
from langdetect import detect
from googletrans import Translator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import aiml  # Import AIML module
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Used for session management

# Load the trained model for gemstone prediction
model = load_model(r'C:\Users\imadt\my_ChatBot\gemstones_CHATBot\models\gemstone_model.h5')

# Class labels for prediction
class_labels = ['Amethyst', 'Diamond', 'Emerald', 'Lapis Lazuli', 'Opal', 'Pearl', 'Quartz Smoky', 'Ruby',
                'Sapphire Yellow', 'Topaz']

# Initialize recognizer and translator for speech-to-text
recognizer = sr.Recognizer()
translator = Translator()

# Create the AIML kernel
kernel = aiml.Kernel()
kernel.learn("C:/Users/imadt/my_ChatBot/gemstones_CHATBot/aiml_files/gemstones.aiml")

# Load spaCy for lemmatization
nlp = spacy.load("en_core_web_sm")

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Load CSV file for try.py responses
df = pd.read_csv(r'C:\Users\imadt\my_ChatBot\gemstones_CHATBot\data\gems.csv')


# Preprocess the text (tokenization, remove punctuation, stopwords, lemmatization)
def preprocess_text(text):
    text = text.translate(str.maketrans("", "", string.punctuation + "0123456789"))
    doc = nlp(text.lower())  # Convert to lowercase and tokenize
    lemmatized_text = " ".join([token.lemma_ for token in doc if not token.is_stop])  # Remove stopwords
    return lemmatized_text


# Apply preprocessing to the questions in your CSV
df['processed_question'] = df['question'].apply(preprocess_text)

# Initialize the TF-IDF Vectorizer with additional parameters
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')

# Fit the TF-IDF model on the processed questions
tfidf_matrix = vectorizer.fit_transform(df['processed_question'])


# Function to get BERT embeddings
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()


# Function to get response based on TF-IDF
def get_response_tfidf(user_input):
    processed_input = preprocess_text(user_input)
    user_input_tfidf = vectorizer.transform([processed_input])
    cosine_sim = cosine_similarity(user_input_tfidf, tfidf_matrix)
    most_similar_idx = cosine_sim.argmax()
    similarity_percentage = cosine_sim[0, most_similar_idx] * 100
    similarity_threshold = max(0.5, cosine_sim[0, most_similar_idx] - 0.1)
    if cosine_sim[0, most_similar_idx] > similarity_threshold:
        return df.iloc[most_similar_idx]['answer'], similarity_percentage
    else:
        return "Sorry, I didn't understand that. Can you please ask something else?", 0


# Function to get response based on BERT embeddings
def get_response_bert(user_input):
    user_input_embeddings = get_bert_embeddings(user_input)
    question_embeddings = np.array([get_bert_embeddings(q) for q in df['processed_question']])
    cosine_sim = cosine_similarity([user_input_embeddings], question_embeddings)
    most_similar_idx = cosine_sim.argmax()
    similarity_percentage = cosine_sim[0, most_similar_idx] * 100
    similarity_threshold = max(0.5, cosine_sim[0, most_similar_idx] - 0.1)
    if cosine_sim[0, most_similar_idx] > similarity_threshold:
        return df.iloc[most_similar_idx]['answer'], similarity_percentage
    else:
        return "Sorry, I didn't understand that. Can you please ask something else?", 0


@app.route('/')
def index():
    return render_template('index.html')  # Serve the frontend page


@app.route('/set_language', methods=['POST'])
def set_language():
    language_choice = request.json.get('language')
    session['language'] = language_choice
    return jsonify(success=True)


@app.route('/recognize_speech', methods=['POST'])
def recognize_speech():
    if 'language' not in session:
        return jsonify(error="Language not set.")

    language_code = session['language']

    with sr.Microphone() as source:
        print(f"Listening for speech in {language_code}...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        try:
            user_input = recognizer.recognize_google(audio, language=language_code)
            print(f"Recognized speech: {user_input}")

            detected_language = detect(user_input)
            if detected_language != 'en':
                user_input = translator.translate(user_input, src=detected_language, dest='en').text

            return jsonify(text=user_input)

        except sr.UnknownValueError:
            return jsonify(error="Sorry, I couldn't understand that.")
        except sr.RequestError:
            return jsonify(error="Sorry, there was an issue with the speech service.")


@app.route("/predict", methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file:
        img = Image.open(file)
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        percentages = predictions[0] * 100
        result = {class_labels[i]: round(float(percentages[i]), 2) for i in range(len(class_labels))}

        return jsonify({"predictions": result})


# New route to handle AIML chatbot responses
@app.route("/chat", methods=['POST'])
def chat():
    user_message = request.json.get('message')  # Get the message from the user

    if user_message:
        # Get the response from the AIML engine
        response = kernel.respond(user_message.upper())  # Ensure we use uppercase for AIML matching

        # Debugging: Log AIML response
        print(f"AIML response: {response}")

        # Check if the AIML response is valid (non-empty, non-generic)
        if not response or "I didn't quite catch that" in response or "I don't know" in response:
            # If AIML doesn't give a proper response, use TF-IDF and BERT fallback
            response_tfidf, similarity_tfidf = get_response_tfidf(user_message)
            response_bert, similarity_bert = get_response_bert(user_message)

            # Compare the similarity percentage between TF-IDF and BERT
            if similarity_bert > similarity_tfidf:
                response = response_bert
            else:
                response = response_tfidf

        # Return the final response to the user
        return jsonify({"response": response})

    return jsonify({"error": "No message provided."})


if __name__ == "__main__":
    app.run(debug=True)
