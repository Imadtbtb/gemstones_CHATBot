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


def play_game():
    print("Welcome to Guess the Gem! 🎮💎\n")

    color = input("Enter color (e.g., Red, Purple, Blue): ").strip().capitalize()
    hardness = int(input("Enter hardness (1-10): "))
    origin = input("Enter origin: ").strip().capitalize()
    clarity = input("Enter clarity (Clear/Opaque): ").strip().capitalize()

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

    if confidence < 60:
        print("\nHmm, I'm not too sure...")
        answer = input("Is the gem typically used in engagement rings? (yes/no) ").lower()
        if answer == 'yes':
            total_scores['Diamond'] += 0.5
            total_scores['Sapphire'] += 0.3
            best_gem = max(total_scores, key=total_scores.get)
            confidence = get_confidence(total_scores)

    print(f"\n💎 I'm {confidence:.1f}% confident your gem is {best_gem}!")


if __name__ == "__main__":
    play_game()
