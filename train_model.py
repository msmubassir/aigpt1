import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np

# ১. ডেটা লোড এবং প্রিপ্রসেসিং
texts = ["This is a positive sentence.", "This is a negative sentence."]  # উদাহরণ
labels = [1, 0]  # 1 = positive, 0 = negative

# টোকেনাইজেশন
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
data = pad_sequences(sequences, maxlen=100)

# ডেটাকে ট্রেনিং ও টেস্টিং ডেটায় ভাগ করা
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# ২. মডেল তৈরি
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=64, input_length=100),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')  # বাইনরি আউটপুট
])

# ৩. মডেল কম্পাইল
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# ৪. মডেল ট্রেনিং
model.fit(X_train, np.array(y_train), epochs=10, batch_size=32)

# ৫. মডেল মূল্যায়ন
loss, accuracy = model.evaluate(X_test, np.array(y_test))
print(f'Accuracy: {accuracy * 100}%')

# মডেল সেভ করা
model.save('model.h5')
