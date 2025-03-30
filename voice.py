import speech_recognition as sr
from langdetect import detect
from googletrans import Translator

recognizer = sr.Recognizer()
translator = Translator()

def get_language_choice():
    choice = input("Enter the number corresponding to your language (1 for English, 2 for French, 3 for Arabic): ")

    if choice == '1':
        return 'en'
    elif choice == '2':
        return 'fr'
    elif choice == '3':
        return 'ar'
    else:
        print("Invalid choice. Defaulting to English.")
        return 'en'


def listen_to_audio(language_code):
    with sr.Microphone() as source:
        print(f"Listening for your question in {language_code}...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            user_input = recognizer.recognize_google(audio, language=language_code)
            print(f"Original Input: {user_input}")
            return user_input
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand that."
        except sr.RequestError:
            return "Sorry, I'm having trouble connecting to the speech service."


def detect_and_translate(user_input):
    detected_language = detect(user_input)
    print(f"Detected Language: {detected_language}")

    if detected_language != 'en':
        translated_text = translator.translate(user_input, src=detected_language, dest='en').text
        print(f"Translated to English: {translated_text}")
        return translated_text
    else:
        # If it's already in English, return as is
        return user_input


language_code = get_language_choice()

user_input = listen_to_audio(language_code)
if user_input:
    translated_input = detect_and_translate(user_input)
    print(f"Final Text (in English): {translated_input}")