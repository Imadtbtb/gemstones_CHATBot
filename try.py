import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

# Load the CSV file from the specified path
df = pd.read_csv(r'C:\Users\imadt\my_ChatBot\gemstones_CHATBot\data\gems.csv')

# Load spaCy for lemmatization
nlp = spacy.load("en_core_web_sm")

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def preprocess_text(text):
    text = text.translate(str.maketrans("", "", string.punctuation + "0123456789"))

    doc = nlp(text.lower())
    lemmatized_text = " ".join([token.lemma_ for token in doc if not token.is_stop])  # Remove stopwords
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

while True:
    # Get user input from the console
    user_input = input("Please ask a question: ")

    # Exit condition for the loop
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break

    # Get responses from both methods
    response_tfidf, similarity_tfidf = get_response_tfidf(user_input)
    response_bert, similarity_bert = get_response_bert(user_input)

    # Print both results
    print("\nResponse using TF-IDF: ")
    print(response_tfidf)
    print(f"Similarity: {similarity_tfidf:.2f}%")

    print("\nResponse using BERT: ")
    print(response_bert)
    print(f"Similarity: {similarity_bert:.2f}%")
