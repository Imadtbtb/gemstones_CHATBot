import random
from nltk.metrics import edit_distance

gemstones = {
    'Amethyst': {'color': 'Purple', 'hardness': 7, 'clarity': 'Clear', 'origin': 'Brazil'},
    'Diamond': {'color': 'Clear', 'hardness': 10, 'clarity': 'Clear', 'origin': 'South Africa'},
    'Emerald': {'color': 'Green', 'hardness': 8, 'clarity': 'Clear', 'origin': 'Colombia'},
    'Opal': {'color': 'Multicolor', 'hardness': 6, 'clarity': 'Opaque', 'origin': 'Australia'},
    'Ruby': {'color': 'Red', 'hardness': 9, 'clarity': 'Clear', 'origin': 'Myanmar'},
    'Sapphire': {'color': 'Blue', 'hardness': 9, 'clarity': 'Clear', 'origin': 'Sri Lanka'}
}

def correct_spelling(user_input, correct_options):
    min_distance = float('inf')
    best_match = user_input
    for option in correct_options:
        distance = edit_distance(user_input.lower(), option.lower())
        if distance < min_distance:
            min_distance = distance
            best_match = option
    return best_match

def fuzzy_match(user_answer, gem_answer):
    distance = edit_distance(user_answer.lower(), gem_answer.lower())
    max_length = max(len(user_answer), len(gem_answer))
    similarity_score = 1 - (distance / max_length)
    return similarity_score

def hardness_distance(user_hardness, gem_hardness):
    return 1 - (abs(user_hardness - gem_hardness) / 10)

backup_questions = [
    "Can you describe the gemstone's texture? (e.g., smooth, rough, shiny, matte)",
    "What is the gemstone's shape? (e.g., round, oval, square, heart-shaped)",
    "Does the gemstone have any special markings or patterns? (e.g., stripes, speckles, plain)",
    "Is the gemstone translucent or opaque?",
    "What is the gemstone's size? (e.g., small, medium, large)"
]

def ask_questions():
    print("Welcome to 'Guess the Gem'!")
    print("Please think of a gemstone from the following list:")
    print("Amethyst, Diamond, Emerald, Opal, Ruby, Sapphire")

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
    extra_question_used = False

    for i, question in enumerate(questions):
        print(question)
        user_answer = input("Your answer: ").strip()

        if user_answer.lower() == "i don't know":
            print(random.choice(backup_questions))
            user_answer = input("Your answer: ").strip()
            extra_question_used = True

        if i == 0:
            user_answer = correct_spelling(user_answer, colors)
        elif i == 1:
            while not user_answer.isdigit() or not (1 <= int(user_answer) <= 10):
                user_answer = input("Please enter a hardness value between 1 and 10: ").strip()
            user_answer = int(user_answer)
        elif i == 2:
            user_answer = correct_spelling(user_answer, origins)
        elif i == 3:
            user_answer = correct_spelling(user_answer, clarity)

        print(f"Did you mean: {user_answer}?")
        user_answers.append(user_answer)

    return user_answers, extra_question_used

def guess_gem(user_answers):
    gem_scores = {}

    for gem, attributes in gemstones.items():
        score = 0
        score += fuzzy_match(user_answers[0], attributes['color'])
        score += hardness_distance(user_answers[1], attributes['hardness'])
        score += fuzzy_match(user_answers[2], attributes['origin'])
        score += fuzzy_match(user_answers[3], attributes['clarity'])

        gem_scores[gem] = score

    guessed_gem = max(gem_scores, key=gem_scores.get)
    return guessed_gem, gem_scores[guessed_gem]

def play_game():
    while True:
        user_answers, extra_question_used = ask_questions()

        guessed_gem, match_score = guess_gem(user_answers)

        print(f"My guess is: {guessed_gem}")
        print(f"Match confidence (fuzziness): {match_score * 100}%")

        response = input("Is this correct? (yes/no)\nYour answer: ")

        if response.lower() == "yes":
            print("Hooray! I guessed it right!")
        else:
            print("Oops! Let's start again.")

        play_again = input("Do you want to play again? (yes/no)\nYour answer: ")
        if play_again.lower() != "yes":
            print("Thanks for playing!")
            break

if __name__ == "__main__":
    play_game()
