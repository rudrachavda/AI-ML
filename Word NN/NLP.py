from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

def create_nlp_model(tokenizer, padded_input_sequences, padded_label_sequences):
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 100
    max_sequence_length = padded_input_sequences.shape[1]  # Get the length of the input sequences
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))
    model.add(LSTM(units=64))
    model.add(Dense(units=vocab_size, activation='softmax'))
    return model
