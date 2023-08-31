import tensorflow as tf
import numpy as np

# Get user input
user_input = input("Enter a word: ").lower()

# Create a vocabulary of characters
vocab = sorted(set(user_input))
char_to_idx = {char: idx for idx, char in enumerate(vocab)}
idx_to_char = np.array(vocab)

# Convert input word to a sequence of character indices
input_sequence = [char_to_idx[char] for char in user_input]

# Create input and target sequences for training
input_data = input_sequence[:-1]
target_data = input_sequence[1:]

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(vocab), output_dim=8, input_length=1),
    tf.keras.layers.LSTM(16, return_sequences=True),
    tf.keras.layers.Dense(len(vocab), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(np.array(input_data), np.array(target_data), epochs=50)

# Generate predictions
start_char = user_input[-1]
start_idx = char_to_idx[start_char]
seed = np.array([[start_idx]])
predicted_idx = model.predict_classes(seed)[0]
predicted_char = idx_to_char[predicted_idx]

print("Input:", user_input)
print("Predicted Next Character:", predicted_char)
