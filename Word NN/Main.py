import json
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
from NLP import create_nlp_model

model = load_model('model.h5')
current_dir = os.getcwd()
file_path = os.path.join(current_dir, "Word NN", "storage.json")

# Load words and definitions from storage.json file
definitions = {}
with open(file_path, 'r') as file:
    definitions = json.load(file)

# Convert definitions to lists of inputs and labels
inputs = list(definitions.keys())
labels = list(definitions.values())

# Convert inputs and labels to sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(inputs + labels)
input_sequences = tokenizer.texts_to_sequences(inputs)
label_sequences = tokenizer.texts_to_sequences(labels)

# Calculate the max sequence length
max_sequence_length = max(len(seq) for seq in input_sequences)

# Pad the input and label sequences
padded_input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
padded_label_sequences = pad_sequences(label_sequences, maxlen=max_sequence_length, padding='post')

# Load the pre-trained model
model = load_model('model.h5')

# Test input loop
while True:
    test_input = input("Enter a test input (or 'q' to quit): ")
    if test_input == 'q':
        break

    test_sequence = tokenizer.texts_to_sequences([test_input])
    padded_test_sequence = pad_sequences(test_sequence, maxlen=max_sequence_length, padding='post')
    predicted_sequence = model.predict(padded_test_sequence)
    predicted_word = tokenizer.index_word[np.argmax(predicted_sequence)]

    print("Predicted word:", predicted_word)
