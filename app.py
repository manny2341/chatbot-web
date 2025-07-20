import json, pickle, random, re
import numpy as np
from flask import Flask, render_template, request, jsonify, session
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)
app.secret_key = 'chatbot-secret-key-2024'

lemmatizer = WordNetLemmatizer()

# Load trained assets
intents  = json.loads(open('intents.json').read())
words    = pickle.load(open('words.pkl', 'rb'))
classes  = pickle.load(open('classes.pkl', 'rb'))
model    = load_model('chatbot_model.keras')


def clean_sentence(sentence):
    tokens = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(w.lower()) for w in tokens]


def bag_of_words(sentence):
    token_words = clean_sentence(sentence)
    bag = [1 if w in token_words else 0 for w in words]
    return np.array(bag)


def predict_intent(sentence):
    bow = bag_of_words(sentence)
    result = model.predict(np.array([bow]), verbose=0)[0]
    best_idx = int(np.argmax(result))
    confidence = float(result[best_idx])
    return classes[best_idx], confidence


def extract_name(message):
    """Try to extract a name from 'my name is X' style messages."""
    patterns = [
        r'my name is (\w+)',
        r"i am called (\w+)",
        r"call me (\w+)",
        r"you can call me (\w+)",
        r"i go by (\w+)",
        r"they call me (\w+)",
        r"people call me (\w+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, message.lower())
        if match:
            return match.group(1).capitalize()
    return None


def get_response(tag, user_message):
    # Check if user is telling us their name
    if tag == 'introduce_name':
        name = extract_name(user_message)
        if name:
            session['user_name'] = name
            responses = [
                f"Nice to meet you, {name}! How can I help you?",
                f"Great name, {name}! What can I do for you?",
                f"Hello {name}! I will remember that. What is on your mind?"
            ]
            return random.choice(responses)

    # Personalize response with name if we know it
    user_name = session.get('user_name', '')

    for intent in intents['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            # Replace <HUMAN> placeholder with known name or "friend"
            if '<HUMAN>' in response:
                response = response.replace('<HUMAN>', user_name if user_name else 'friend')
            # Occasionally add name at the start for personalization
            if user_name and random.random() < 0.25 and tag in ['greeting', 'CourtesyGreeting', 'motivation']:
                response = f"{user_name}, {response[0].lower()}{response[1:]}"
            return response

    return "I am not sure I understand. Could you rephrase that?"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({'response': 'Please type something!'})

    tag, confidence = predict_intent(user_message)

    if confidence < 0.25:
        response = "I am not sure I understand. Could you rephrase that?"
    else:
        response = get_response(tag, user_message)

    return jsonify({
        'response': response,
        'intent': tag,
        'confidence': round(confidence * 100, 1)
    })


@app.route('/clear', methods=['POST'])
def clear():
    session.pop('user_name', None)
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(debug=True, port=5005)
