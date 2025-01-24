import aiml

# Create a Kernel (this is the AIML engine)
kernel = aiml.Kernel()

# Load your AIML file with the full path
kernel.learn("C:/Users/imadt/my_ChatBot/gemstones_CHATBot/aiml_files/gemstones.aiml")

# Test basic interaction
while True:
    user_input = input("You: ").upper()  # Convert to uppercase for pattern matching
    if user_input.lower() == 'quit':
        break
    response = kernel.respond(user_input)
    print("Bot:", response)
