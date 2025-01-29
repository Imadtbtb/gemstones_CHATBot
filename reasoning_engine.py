import os
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from difflib import get_close_matches

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

file_path = r'C:\Users\imadt\my_ChatBot\gemstones_CHATBot\data\logic.txt'

# Custom color synonyms to enhance WordNet data
color_synonyms = {
    'green': ['green', 'emerald', 'lime', 'olive', 'jade', 'chartreuse'],
    'red': ['red', 'ruby', 'crimson', 'scarlet', 'vermilion'],
    'blue': ['blue', 'sapphire', 'azure', 'cobalt', 'navy'],
    # Add more color mappings as needed
}


def load_knowledge_base(file_path):
    kb = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            logic_data = file.readlines()
        for line in logic_data:
            if line.strip():
                fact = line.strip().lower()
                try:
                    key, value = fact.split(" -> ")
                    if key not in kb:
                        kb[key] = []
                    kb[key].append(value)
                except ValueError:
                    print(f"Skipping invalid line: {line.strip()}")
    return kb


def check_contradictions(kb, user_input):
    try:
        user_input = user_input.lower()
        key, value = user_input.split(" -> ")
        value_inner = value.split('(')[1].split(')')[0] if '(' in value else value
        value_synonyms = get_synonyms(value_inner)
        check_values = [value] + [f"gemstone({syn})" for syn in value_synonyms]

        if key in kb:
            if any(cv in kb[key] for cv in check_values):
                return "This fact is already known to me."
            negated_value = f"gemstone(not {value_inner})"
            if negated_value in kb[key]:
                return "This seems to contradict what I know."
        return "No contradictions found."
    except Exception as e:
        return f"Error while processing the input: {e}"


def tokenize_and_tag(input_text):
    tokens = word_tokenize(input_text.lower())
    tagged_tokens = pos_tag(tokens)
    return tagged_tokens


def get_synonyms(word):
    synonyms = set()
    # Add custom color synonyms
    word_lower = word.lower()
    if word_lower in color_synonyms:
        synonyms.update(color_synonyms[word_lower])
    # Add WordNet synonyms, hyponyms, and hypernyms
    for syn in wordnet.synsets(word_lower):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ').lower()
            if synonym != word_lower:
                synonyms.add(synonym)
        # Hyponyms (more specific terms)
        for hypo in syn.hyponyms():
            for lemma in hypo.lemmas():
                synonyms.add(lemma.name().replace('_', ' ').lower())
        # Hypernyms (more general terms)
        for hyper in syn.hypernyms():
            for lemma in hyper.lemmas():
                synonyms.add(lemma.name().replace('_', ' ').lower())
    return list(synonyms)


def find_closest_key(user_gem, existing_keys):
    # Extract gemstone names from existing keys (e.g., "gemstone(emerald)" -> "emerald")
    gem_names = [key.split('(')[1].split(')')[0] for key in existing_keys]
    # Find closest match to user_gem
    matches = get_close_matches(user_gem, gem_names, n=1, cutoff=0.7)
    if matches:
        return f"gemstone({matches[0]})"
    return None


def handle_user_input(kb, user_input):
    user_input = user_input.strip().lower()
    tagged_input = tokenize_and_tag(user_input)
    print(f"Tagged input: {tagged_input}")

    if "i know that" in user_input and "is" in user_input:
        try:
            parts = user_input.split("i know that")[1].strip().split("is")
            fact_key_part = parts[0].strip()
            fact_value_part = parts[1].strip()
            fact = f"gemstone({fact_key_part}) -> gemstone({fact_value_part})"
            feedback = check_contradictions(kb, fact)

            if feedback == "No contradictions found.":
                key, value = fact.split(" -> ")
                if key not in kb:
                    kb[key] = []
                kb[key].append(value)
                return "Got it! I've updated my knowledge with this new information."
            else:
                return feedback
        except Exception as e:
            return f"Error while processing the input pattern: {e}"

    elif "check" in user_input and "is" in user_input:
        try:
            parts = user_input.split("check")[1].strip().split("is")
            fact_key_part = parts[0].strip()
            fact_value_part = parts[1].strip().lower()
            original_key = f"gemstone({fact_key_part})"

            # Check for typos in the key
            if original_key not in kb:
                closest_key = find_closest_key(fact_key_part, kb.keys())
                key = closest_key if closest_key else original_key
            else:
                key = original_key

            if key in kb:
                # Check each stored value's synonyms
                for stored_value in kb[key]:
                    stored_inner = stored_value.split('(')[1].split(')')[0].lower()
                    stored_syns = get_synonyms(stored_inner)
                    if fact_value_part == stored_inner or fact_value_part in stored_syns:
                        return "Yes, that’s correct. I already know that."
                return "I don't have that information. It seems to contradict what I know."
            else:
                return "I don't have any information about this gemstone."
        except Exception as e:
            return f"Error while processing the 'Check' input: {e}"

    else:
        return "I didn't quite get that. Please use 'I know that ... is ...' or 'Check ... is ...'."


def save_knowledge_base(kb, file_path):
    with open(file_path, 'w') as file:
        for key, values in kb.items():
            for value in values:
                file.write(f"{key} -> {value}\n")


def chatbot():
    kb = load_knowledge_base(file_path)
    print("Gemstone ChatBot: Hello! How can I assist you today?")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            print("Gemstone ChatBot: Goodbye! It was a pleasure assisting you.")
            break
        response = handle_user_input(kb, user_input)
        print(f"Gemstone ChatBot: {response}")
        save_knowledge_base(kb, file_path)


if __name__ == "__main__":
    chatbot()