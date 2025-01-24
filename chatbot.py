import aiml
from flask import Flask, render_template, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Create the AIML kernel
kernel = aiml.Kernel()

# Load your AIML file
kernel.learn("aiml_files/gemstones.aiml")  # Adjust path if necessary

@app.route("/")
def index():
    return render_template('index.html')  # Render the chat interface

@app.route("/ask", methods=['POST'])
def ask_bot():
    user_input = request.json.get('message')  # Get message from frontend
    response = kernel.respond(user_input)    # Get the AIML response
    return jsonify({"response": response})   # Return the response as JSON

if __name__ == "__main__":
    app.run(debug=True)
