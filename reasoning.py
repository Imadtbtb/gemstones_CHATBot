import re
import os
from nltk.sem.logic import Expression
from nltk.inference import ResolutionProver

read_expr = Expression.fromstring

kb_file_path = r"C:\Users\imadt\my_ChatBot\gemstones_CHATBot\data\kb.txt"


def load_kb():
    """
    Load facts from the knowledge base file into a set of NLTK expressions.
    Each fact in the file is expected to be on its own line, e.g., European(Tim)
    """
    kb = set()
    if os.path.exists(kb_file_path):
        with open(kb_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        expr = read_expr(line)
                        kb.add(expr)
                    except Exception as e:
                        print(f"Error parsing the fact '{line}': {e}")
    return kb


def save_kb(kb):
    """
    Save the current set of facts back to the knowledge base file.
    Each fact is written in its string representation on a new line.
    """
    with open(kb_file_path, 'w') as f:
        for fact in kb:
            f.write(str(fact) + "\n")


def check_kb_consistency(kb):
    """
    Check the KB for internal contradictions.
    If a fact and its negation both exist, print a warning message.
    Returns True if KB is consistent, False otherwise.
    """
    consistent = True
    for fact in kb:
        fact_str = str(fact)
        # If the fact is negated, check for its positive form.
        if fact_str.startswith("~"):
            pos_fact_str = fact_str[1:]
            try:
                pos_fact = read_expr(pos_fact_str)
            except Exception:
                continue
            if pos_fact in kb:
                print(f"Contradiction detected in KB: {fact} and {pos_fact}")
                consistent = False
        else:
            neg_fact_str = "~" + fact_str
            try:
                neg_fact = read_expr(neg_fact_str)
            except Exception:
                continue
            if neg_fact in kb:
                print(f"Contradiction detected in KB: {fact} and {neg_fact}")
                consistent = False
    return consistent


def entails(hypothesis, premises):
    """
    Uses NLTK's ResolutionProver to check if the premises entail the hypothesis.
    Returns True if the hypothesis is entailed, False otherwise.
    """
    prover = ResolutionProver()
    return prover.prove(hypothesis, premises)


def process_know_command(command, kb):
    """
    Process commands of the form:
      "I know that <Subject> is <Predicate>"
    Before adding the fact, this function checks if the negated fact (i.e.,
    "I know that <Subject> is not <Predicate>") exists in the KB.
    If the fact exists in the KB, it responds with "I know that."
    If the negated fact exists, it returns a contradiction message.
    Otherwise, it adds the fact to the KB and responds with "Ok, I will remember that."
    """
    pattern = r"^I know that (\w+) is (\w+)$"
    match = re.match(pattern, command)
    if not match:
        return "Invalid format. Please use: I know that X is Y"
    subject = match.group(1)
    predicate = match.group(2)
    fact_str = f"{predicate}({subject})"
    try:
        fact_expr = read_expr(fact_str)
    except Exception as e:
        return f"Error parsing the fact: {fact_str}. {e}"
    # Check for contradiction if the negated fact exists.
    try:
        neg_fact_expr = read_expr("~" + fact_str)
        if neg_fact_expr in kb:
            return f"Contradiction detected: I already know that {subject} is not {predicate}."
    except Exception:
        pass
    if fact_expr in kb:
        return "I know that."
    else:
        kb.add(fact_expr)
        save_kb(kb)
        return f"Ok, I will remember that {subject} is {predicate}."


def process_know_not_command(command, kb):
    """
    Process commands of the form:
      "I know that <Subject> is not <Predicate>"
    Before adding the negated fact, this function checks if the positive fact (i.e.,
    "I know that <Subject> is <Predicate>") exists in the KB.
    If the negated fact exists in the KB, it responds with "I know that."
    If the positive fact exists, it returns a contradiction message.
    Otherwise, it adds the negated fact to the KB and responds with "Ok, I will remember that."
    """
    pattern = r"^I know that (\w+) is not (\w+)$"
    match = re.match(pattern, command)
    if not match:
        return "Invalid format. Please use: I know that X is not Y"
    subject = match.group(1)
    predicate = match.group(2)
    fact_str = f"{predicate}({subject})"
    neg_fact_str = f"~{fact_str}"
    try:
        neg_fact_expr = read_expr(neg_fact_str)
    except Exception as e:
        return f"Error parsing the fact: {neg_fact_str}. {e}"
    # Check for contradiction if the positive fact exists.
    try:
        pos_fact_expr = read_expr(fact_str)
        if pos_fact_expr in kb:
            return f"Contradiction detected: I already know that {subject} is {predicate}."
    except Exception:
        pass
    if neg_fact_expr in kb:
        return "I know that."
    else:
        kb.add(neg_fact_expr)
        save_kb(kb)
        return f"Ok, I will remember that {subject} is not {predicate}."


def process_check_command(command, kb):
    """
    Process commands of the form:
      "Check that <Subject> is <Predicate>"
    This function uses NLTK's ResolutionProver to check the query:
      - If the KB entails the fact, it returns "Correct".
      - If the KB entails the negation of the fact, it returns "Incorrect".
      - Otherwise, it returns "I don't know."
    """
    pattern = r"^Check that (\w+) is (\w+)$"
    match = re.match(pattern, command)
    if not match:
        return "Invalid format. Please use: Check that X is Y"
    subject = match.group(1)
    predicate = match.group(2)
    fact_str = f"{predicate}({subject})"
    try:
        fact_expr = read_expr(fact_str)
        neg_fact_expr = read_expr("~" + fact_str)
    except Exception as e:
        return f"Error parsing the fact: {fact_str}. {e}"

    premises = list(kb)

    if entails(fact_expr, premises):
        return "Correct"
    elif entails(neg_fact_expr, premises):
        return "Incorrect"
    else:
        return "I don't know."


def main():
    kb = load_kb()
    if not check_kb_consistency(kb):
        print("Warning: Inconsistencies detected in the knowledge base.")
    print("Type your commands. Type 'exit' to quit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break
        elif user_input.startswith("I know that"):
            if " is not " in user_input:
                response = process_know_not_command(user_input, kb)
            else:
                response = process_know_command(user_input, kb)
        elif user_input.startswith("Check that"):
            response = process_check_command(user_input, kb)
        else:
            response = ("Command not recognized. Please use 'I know that X is Y', " +
                        "'I know that X is not Y', or 'Check that X is Y'.")
        print("Bot:", response)


if __name__ == '__main__':
    main()
