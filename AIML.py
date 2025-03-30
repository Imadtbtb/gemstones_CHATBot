import aiml

kernel = aiml.Kernel()

# Load the AIML file
kernel.learn("C:/Users/imadt/my_ChatBot/gemstones_CHATBot/aiml_files/gemstones.aiml")

while True:
    user_input = input("You: ").strip().upper()

    if user_input.lower() == 'quit':
        break

    print("User input:", repr(user_input))

    response = kernel.respond(user_input)

    print("Kernel response:", repr(response))

    if not response:
        response = "I'm sorry, I didn't quite catch that. Can you ask about gemstones or their properties?"

    print("Bot:", response)
