"""
Train the chatbot model from intents.json and save:
  - words.pkl
  - classes.pkl
  - chatbot_model.keras
Run once before starting the web app.
"""
import ssl, json, pickle, random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Fix SSL for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words, classes, documents = [], [], []
ignore_letters = ['?', '!', '.', ',', ':']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = sorted(set(
    lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters
))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Build training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in doc[0]]
    bag = [1 if w in pattern_words else 0 for w in words]
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Build model
model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax'),
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=500, batch_size=5, verbose=1)

model.save('chatbot_model.keras')
print("\n✅ Model trained and saved as chatbot_model.keras")
print(f"   Words: {len(words)}   Classes: {len(classes)}")
