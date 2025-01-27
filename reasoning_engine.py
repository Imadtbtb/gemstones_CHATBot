import os

# Define the path to the logic file
file_path = r'C:\Users\imadt\my_ChatBot\gemstones_CHATBot\data\logic.txt'


# Function to load the knowledge base from the file
def load_knowledge_base(file_path):
    kb = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            logic_data = file.readlines()

        # Add facts to the knowledge base
        for line in logic_data:
            if line.strip():  # Skip empty lines
                fact = line.strip().lower()  # Convert to lowercase for case-insensitive comparison
                key, value = fact.split(" -> ")
                if key not in kb:
                    kb[key] = []
                kb[key].append(value)

    return kb


# Function to check contradictions with the knowledge base
def check_contradictions(kb, user_input):
    try:
        # Convert input to lowercase for case-insensitive processing
        user_input = user_input.lower()

        key, value = user_input.split(" -> ")

        # Check if the fact is already in the knowledge base
        if key in kb and value in kb[key]:
            return "This fact is already known to me."

        # Check for contradictions by checking if the negation of the fact exists
        negated_value = f"not {value}"
        if key in kb and negated_value in kb[key]:
            return "This seems to contradict what I know."

        return "No contradictions found."

    except Exception as e:
        return f"Error while processing the input: {e}"


# Function to handle and process user input patterns like "I know that ... is ..."
def handle_user_input(kb, user_input):
    user_input = user_input.strip().lower()  # Convert input to lowercase

    if "i know that" in user_input and "is" in user_input:
        # Extract the information after "I know that" and before "is"
        try:
            parts = user_input.split("i know that")[1].strip().split("is")
            fact = f"gemstone({parts[0].strip()}) -> gemstone({parts[1].strip()})"
            feedback = check_contradictions(kb, fact)

            if feedback == "No contradictions found.":
                # Add the new fact to the knowledge base
                key, value = fact.split(" -> ")
                if key not in kb:
                    kb[key] = []
                kb[key].append(value)
                return f"Got it! I've updated my knowledge with this new information."

            else:
                return feedback

        except Exception as e:
            return f"Error while processing the input pattern: {e}"

    elif "check" in user_input and "is" in user_input:
        # Handle "Check ... is ..." query
        try:
            parts = user_input.split("check")[1].strip().split("is")
            fact = f"gemstone({parts[0].strip()}) -> gemstone({parts[1].strip()})"

            # Check if the fact exists in the knowledge base
            key, value = fact.split(" -> ")
            if key in kb and value in kb[key]:
                return "Yes, that’s correct. I already know that."
            else:
                return "I don't have that information. It seems to contradict what I know."

        except Exception as e:
            return f"Error while processing the 'Check' input: {e}"

    else:
        return "I didn't quite get that. Please use 'I know that ... is ...' or 'Check ... is ...'."


# Function to save the updated knowledge base to the file
def save_knowledge_base(kb, file_path):
    with open(file_path, 'w') as file:
        for key, values in kb.items():
            for value in values:
                file.write(f"{key} -> {value}\n")


# Main function to interact with the chatbot and handle user input
def chatbot():
    # Load the existing knowledge base from the file
    kb = load_knowledge_base(file_path)

    print("Gemstone ChatBot: Hello! How can I assist you today?")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == 'exit':
            print("Gemstone ChatBot: Goodbye! It was a pleasure assisting you.")
            break

        # Process user input and handle logic
        response = handle_user_input(kb, user_input)
        print(f"Gemstone ChatBot: {response}")

        # Save the updated knowledge base
        save_knowledge_base(kb, file_path)


# Start the chatbot
if __name__ == "__main__":
    chatbot()
