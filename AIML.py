import aiml

# Create a Kernel (this is the AIML engine)
kernel = aiml.Kernel()

# Load the AIML file
kernel.learn("C:/Users/imadt/my_ChatBot/gemstones_CHATBot/aiml_files/gemstones.aiml")

# Test basic interaction
while True:
    user_input = input("You: ").strip().upper()  # Strip any unwanted spaces and use uppercase for pattern matching

    # Exit condition for 'quit' command
    if user_input.lower() == 'quit':
        break

    # Debugging step to print the raw input
    print("User input:", repr(user_input))  # Shows input with any hidden characters

    # Get response from AIML engine
    response = kernel.respond(user_input)  # Kernel will try to match the pattern

    # Debugging: check kernel's response
    print("Kernel response:", repr(response))  # Shows the response from the kernel

    # If no response, print a fallback message
    if not response:
        response = "I'm sorry, I didn't quite catch that. Can you ask about gemstones or their properties?"

    print("Bot:", response)
