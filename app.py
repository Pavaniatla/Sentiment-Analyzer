from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model_utils import create_model 

app = Flask(__name__)
max_len = 300 
sentiment_mapping = {
    "class_1": "Very Negative",
    "class_2": "Negative",
    "class_3": "Neutral",
    "class_4": "Positive",
    "class_5": "Very Positive"
}

def load_model_and_resources(model_type='LSTM', input_length=100, num_classes=5, max_vocab_size=5000):
    with open(f'models/{model_type}_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    with open(f'models/{model_type}_label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    vocab_size = min(len(tokenizer.word_index) + 1, max_vocab_size)

    model = create_model(input_len=input_length, model_type=model_type, vocab_size=vocab_size, num_classes=num_classes)

    model.build(input_shape=(None, input_length))

    model.load_weights(f'models/{model_type}_weights.weights.h5')

    return model, tokenizer, label_encoder

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    input_text = ""
    model_type = "LSTM"  
    sentiment_map = {
        "class_1": "Very Negative",
        "class_2": "Negative",
        "class_3": "Neutral",
        "class_4": "Positive",
        "class_5": "Very Positive"
    }

    if request.method == 'POST':
        input_text = request.form['input_text']
        model_type = request.form['model_type']

        model, tokenizer, label_encoder = load_model_and_resources(model_type)
        sequence = tokenizer.texts_to_sequences([input_text])
        padded = pad_sequences(sequence, maxlen=max_len)
        prediction = model.predict(padded)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
        sentiment = sentiment_map.get(predicted_label[0], "Unknown")

    return render_template('index.html', sentiment=sentiment, input_text=input_text, model_type=model_type)



if __name__ == '__main__':
    app.run(debug=True)
