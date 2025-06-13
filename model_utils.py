from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SimpleRNN, Bidirectional, GRU

def create_model(input_len, model_type='LSTM', vocab_size=5000, num_classes=5):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=input_len))
    
    if model_type == 'RNN':
        model.add(SimpleRNN(128))
    elif model_type == 'LSTM':
        model.add(LSTM(128))
    elif model_type == 'BiLSTM':
        model.add(Bidirectional(LSTM(128)))
    elif model_type == 'GRU':
        model.add(GRU(128))
    else:
        raise ValueError("Invalid model_type")
        
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
