from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Flask অ্যাপ তৈরি
app = Flask(__name__)

# মডেল লোড
model = tf.keras.models.load_model('model.h5')

# টোকেনাইজার লোড
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(["This is a sample sentence"])  # ডেটার সাথে মিলে যাবে

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json.get('text')
    
    # টেক্সট প্রিপ্রসেস করা
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100)
    
    # প্রেডিকশন
    prediction = model.predict(padded)
    
    # প্রেডিকশন ফলাফল পাঠানো
    return jsonify({'prediction': 'positive' if prediction[0] > 0.5 else 'negative'})

if __name__ == '__main__':
    app.run(debug=True)
