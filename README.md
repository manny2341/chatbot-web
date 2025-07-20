# 🤖 AI Chatbot — Web Version

A neural network chatbot rebuilt as a full web app with a modern chat bubble interface. Originally a terminal script, now runs in the browser with real-time responses, a typing indicator, and a clean chat UI.

## Features

- Real-time chat interface (like WhatsApp / iMessage)
- Typing indicator animation while Bot is "thinking"
- 18 conversation intents — greetings, questions, smalltalk, and more
- Neural network backend trained with TensorFlow/Keras
- Confidence threshold — falls back gracefully on unknown input

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Web Framework | Flask (Python) |
| Deep Learning | TensorFlow / Keras |
| NLP | NLTK (tokenization + lemmatization) |
| Frontend | HTML, CSS, Vanilla JavaScript |

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/manny2341/chatbot-web.git
cd chatbot-web
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the model** *(only needed once)*
```bash
python3 train.py
```

**4. Start the app**
```bash
python3 app.py
```

**5. Open in browser**
```
http://127.0.0.1:5005
```

## How It Works

1. User types a message in the chat input
2. Message is sent to Flask via a POST request (`/chat`)
3. NLTK tokenizes and lemmatizes the words
4. A **bag-of-words** vector is created from the message
5. The neural network predicts which **intent** the message belongs to
6. A random response for that intent is returned and displayed

## Project Structure

```
chatbot-web/
├── app.py            # Flask server + prediction logic
├── train.py          # Train and save the model
├── intents.json      # All conversation intents and responses
├── templates/
│   └── index.html    # Chat UI
├── static/
│   └── style.css     # Styling
└── requirements.txt
```

## Sample Conversations

| You say | Bot replies |
|---------|------------|
| Hello | Hi there! / Good to see you again! |
| What is your name? | I'm Bot! |
| How old are you? | I am 18 years old! |
| Goodbye | Bye! Come back again soon. |
| How are you? | Hello, I am great, how are you? |

## Author

[@manny2341](https://github.com/manny2341)
