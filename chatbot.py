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
import random
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
# Import your game functions from game.py
from game import ask_questions, guess_gem, correct_spelling
import re
import os
from nltk.sem.logic import Expression
from reasoning import process_know_not_command,process_check_command,process_know_command,load_kb

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Used for session management
read_expr = Expression.fromstring
# Define your knowledge base file path.
kb_file_path = r"C:\Users\imadt\my_ChatBot\gemstones_CHATBot\data\kb.txt"

# ------------------- Gemstone Prediction Model -------------------
# Load the trained model for gemstone prediction
model = load_model(r'C:\Users\imadt\my_ChatBot\gemstones_CHATBot\models\gemstone_model.h5')

# Class labels for prediction
class_labels = ['Amethyst', 'Diamond', 'Emerald', 'Lapis Lazuli', 'Opal', 'Pearl', 'Quartz Smoky', 'Ruby',
                'Sapphire Yellow', 'Topaz']

# ------------------- Speech Recognition and Translation -------------------
recognizer = sr.Recognizer()
translator = Translator()

# ------------------- AIML Chatbot -------------------
kernel = aiml.Kernel()
kernel.learn(r"C:\Users\imadt\my_ChatBot\gemstones_CHATBot\aiml_files\gemstones.aiml")

# ------------------- spaCy, TF-IDF, and BERT Setup -------------------
nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

df = pd.read_csv(r'C:\Users\imadt\my_ChatBot\gemstones_CHATBot\data\gems.csv')


def preprocess_text(text):
    text = text.translate(str.maketrans("", "", string.punctuation + "0123456789"))
    doc = nlp(text.lower())
    lemmatized_text = " ".join([token.lemma_ for token in doc if not token.is_stop])
    return lemmatized_text


df['processed_question'] = df['question'].apply(preprocess_text)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['processed_question'])


def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()


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


# ------------------- Routes -------------------

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


@app.route("/chat", methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if user_message:
        response = kernel.respond(user_message.upper())
        print(f"AIML response: {response}")
        if not response or "I didn't quite catch that" in response or "I don't know" in response:
            response_tfidf, similarity_tfidf = get_response_tfidf(user_message)
            response_bert, similarity_bert = get_response_bert(user_message)
            if similarity_bert > similarity_tfidf:
                response = response_bert
            else:
                response = response_tfidf
        return jsonify({"response": response})
    return jsonify({"error": "No message provided."})


# ------------------- Interactive Chat Game (Guess the Gem) -------------------
# We'll store active game generator instances here keyed by a unique game id.
games = {}


def game_generator():
    # Welcome messages (as in your console game)
    yield "Welcome to 'Guess the Gem'!"
    yield "Please think of a gemstone from the following list:"
    yield "Amethyst, Diamond, Emerald, Opal, Ruby, Sapphire"

    # Define options and questions (matching your ask_questions function)
    colors = ['Purple', 'Clear', 'Green', 'Multicolor', 'Red', 'Blue']
    origins = ['Brazil', 'South Africa', 'Colombia', 'Australia', 'Myanmar', 'Sri Lanka']
    clarity = ['Clear', 'Opaque']
    questions = [
        "What color is your gemstone? (Options: Purple, Clear, Green, Multicolor, Red, Blue)",
        "How hard is your gemstone? (Enter the hardness on a scale of 1 to 10, with 10 being the hardest)",
        "Where is your gemstone from? (Options: Brazil, South Africa, Colombia, Australia, Myanmar, Sri Lanka)",
        "What is the clarity of your gemstone? (Options: Clear, Opaque)"
    ]
    user_answers = []

    # For each question, yield the question then wait for an answer.
    for i, q in enumerate(questions):
        yield q
        answer = yield "Your answer:"  # Pause for input.

        # If the user types "i don't know", ask a backup question.
        if answer.lower() == "i don't know":
            backup_questions = [
                "Can you describe the gemstone's texture? (e.g., smooth, rough, shiny, matte)",
                "What is the gemstone's shape? (e.g., round, oval, square, heart-shaped)",
                "Does the gemstone have any special markings or patterns? (e.g., stripes, speckles, plain)",
                "Is the gemstone translucent or opaque?",
                "What is the gemstone's size? (e.g., small, medium, large)"
            ]
            backup_q = random.choice(backup_questions)
            yield backup_q
            answer = yield "Your answer:"  # Get backup answer.

        # Process the answer as in your original game logic.
        if i == 0:
            answer = correct_spelling(answer, colors)
        elif i == 1:
            try:
                answer = int(answer)
            except ValueError:
                answer = 5  # Fallback default.
        elif i == 2:
            answer = correct_spelling(answer, origins)
        elif i == 3:
            answer = correct_spelling(answer, clarity)

        yield f"Did you mean: {answer}?"
        user_answers.append(answer)

    # Use guessing logic.
    guessed_gem, match_score = guess_gem(user_answers)
    yield f"My guess is: {guessed_gem}"
    yield f"Match confidence (fuzziness): {match_score * 100}%"
    yield "Is this correct? (yes/no)"
    response = yield "Your answer:"  # Wait for confirmation.
    if response.lower() == "yes":
        yield "Hooray! I guessed it right!"
    else:
        yield "Oops! Let's start again."
    yield "Do you want to play again? (yes/no)"
    play_again = yield "Your answer:"
    if play_again.lower() == "yes":
        yield "Restarting game..."
        # (Here you might reinitialize a new game instance.)
    else:
        yield "Thanks for playing!"


@app.route('/play_game')
def play_game():
    # When a user visits /play_game, create a new game generator instance.
    from uuid import uuid4
    game_id = str(uuid4())
    gen = game_generator()
    games[game_id] = gen
    messages = []
    try:
        # Prime the generator to get the first message.
        msg = next(gen)
        messages.append(msg)
    except StopIteration:
        pass
    return render_template("game.html", game_id=game_id, messages=messages)


@app.route('/next', methods=['POST'])
def next_message():
    game_id = request.form.get("game_id")
    user_input = request.form.get("user_input")
    if game_id not in games:
        return jsonify({"messages": ["Game not found."]})

    gen = games[game_id]
    messages = []
    try:
        msg = gen.send(user_input)
        messages.append(msg)

        while True:
            msg = next(gen)
            messages.append(msg)
            if msg == "Your answer:":
                break
    except StopIteration:

        del games[game_id]
    except Exception as e:
        messages.append("Error: " + str(e))
    return jsonify({"messages": messages})

@app.route('/game2')
def game2():
    return render_template('game2.html')


@app.route('/process_command', methods=['POST'])
def process_command():
    user_input = request.json.get("command")
    if not user_input:
        return jsonify({"error": "No command provided"}), 400

    kb = load_kb()

    if user_input.startswith("I know that"):
        if " is not " in user_input:
            response = process_know_not_command(user_input, kb)
        else:
            response = process_know_command(user_input, kb)
    elif user_input.startswith("Check that"):
        response = process_check_command(user_input, kb)
    else:
        response = ("Command not recognized. Please use 'I know that X is Y', " +
                    "'I know that X is not Y', or 'Check that X is Y'.")

    return jsonify({"response": response})

@app.route('/try')
def try_page():
    return render_template('try.html')


gems = ['Amethyst', 'Diamond', 'Emerald', 'Opal', 'Ruby', 'Sapphire']

color_mapping = {
    'Purple': {'Amethyst': 1.0, 'Sapphire': 0.3},
    'Clear': {'Diamond': 1.0, 'Opal': 0.4},
    'Green': {'Emerald': 1.0},
    'Multicolor': {'Opal': 1.0},
    'Red': {'Ruby': 1.0},
    'Blue': {'Sapphire': 1.0, 'Amethyst': 0.3},
}

hardness_mapping = {
    'Soft': {'Opal': 1.0},
    'Medium': {'Amethyst': 0.8, 'Emerald': 0.9},
    'Hard': {'Diamond': 1.0, 'Ruby': 1.0, 'Sapphire': 1.0},
}

origin_mapping = {
    'Brazil': {'Amethyst': 1.0},
    'South Africa': {'Diamond': 1.0},
    'Colombia': {'Emerald': 1.0},
    'Australia': {'Opal': 1.0},
    'Myanmar': {'Ruby': 1.0},
    'Sri Lanka': {'Sapphire': 1.0},
}

clarity_mapping = {
    'Clear': {'Diamond': 1.0, 'Amethyst': 0.9, 'Ruby': 1.0, 'Sapphire': 1.0, 'Emerald': 0.9},
    'Opaque': {'Opal': 1.0},
}

rules = [
    {'conditions': [('color', 'Blue'), ('hardness', 'Hard')], 'consequent': 'Sapphire'},
    {'conditions': [('color', 'Green'), ('hardness', 'Medium')], 'consequent': 'Emerald'},
    {'conditions': [('color', 'Clear'), ('hardness', 'Hard')], 'consequent': 'Diamond'},
    {'conditions': [('color', 'Multicolor'), ('clarity', 'Opaque')], 'consequent': 'Opal'},
    {'conditions': [('color', 'Red'), ('clarity', 'Clear')], 'consequent': 'Ruby'},
    {'conditions': [('color', 'Purple'), ('hardness', 'Medium')], 'consequent': 'Amethyst'},
]


def fuzzify_hardness(hardness):
    if hardness == 7:
        return {'Medium': 0.7, 'Hard': 0.3}
    if hardness <= 4:
        return {'Soft': 1.0}
    elif 5 <= hardness <= 8:
        return {'Medium': 1.0}
    else:
        return {'Hard': 1.0}


def calculate_scores(inputs):
    scores = {gem: 0.0 for gem in gems}
    for category, mapping in [('color', color_mapping), ('hardness', hardness_mapping), ('origin', origin_mapping),
                              ('clarity', clarity_mapping)]:
        for key, strength in inputs[category].items():
            for gem, value in mapping.get(key, {}).items():
                scores[gem] += strength * value
    return scores


def apply_rules(inputs):
    rule_scores = {gem: 0.0 for gem in gems}
    for rule in rules:
        min_strength = 1.0
        valid = True
        for (var_type, category) in rule['conditions']:
            strength = inputs[var_type].get(category, 0.0)
            min_strength = min(min_strength, strength)
            if strength == 0:
                valid = False
                break
        if valid:
            rule_scores[rule['consequent']] += min_strength
    return rule_scores


def get_confidence(scores):
    total = sum(scores.values())
    return 0.0 if total == 0 else max(scores.values()) / total * 100


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/run-the-game', methods=['POST'])
def run_the_game():
    data = request.json
    color = data.get('color', '').capitalize()
    hardness = int(data.get('hardness', 0))
    origin = data.get('origin', '').capitalize()
    clarity = data.get('clarity', '').capitalize()

    inputs = {
        'color': {color: 1.0},
        'hardness': fuzzify_hardness(hardness),
        'origin': {origin: 1.0},
        'clarity': {clarity: 1.0},
    }

    var_scores = calculate_scores(inputs)
    rule_scores = apply_rules(inputs)
    total_scores = {gem: var_scores[gem] + rule_scores[gem] for gem in gems}

    best_gem = max(total_scores, key=total_scores.get)
    confidence = get_confidence(total_scores)

    return jsonify({'gem': best_gem, 'confidence': f'{confidence:.1f}%'}), 200



# ------------------- Endd of Routes -------------------

if __name__ == "__main__":
    app.run(debug=True)
