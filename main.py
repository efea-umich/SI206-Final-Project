from keras.preprocessing.text import Tokenizer
import nltk
import pandas as pd
import numpy as np
import keras
from tensorflow.python.keras.callbacks import ModelCheckpoint

with open('OnionOrNot.csv', 'r') as hand:
    data = pd.read_csv(hand)
    headlines = data[data['label'] == 1]["text"]


max_words = 75000
tokenizer = Tokenizer()
tokenizer.fit_on_texts(headlines.values)
sequences = tokenizer.texts_to_sequences(headlines.values)
text = [item for sublist in sequences for item in sublist]
vocab_size = len(tokenizer.word_index)

print(text)
print(np.shape(text))

# Training on 19 words to predict the 20th
sentence_len = 20
pred_len = 1
train_len = sentence_len - pred_len
seq = []
# Sliding window to generate train data
for i in range(len(text)-sentence_len):
    seq.append(text[i:i+sentence_len])
# Reverse dictionary to decode tokenized sequences back to words
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

# Each row in seq is a 20 word long window. We append he first 19 words as the input to predict the 20th word
trainX = []
trainy = []
for i in seq:
    trainX.append(i[:train_len])
    trainy.append(i[-1])
# define model
model_2 = keras.models.Sequential([
    keras.layers.Embedding(vocab_size+1, 50, input_length=train_len),
    keras.layers.LSTM(100, return_sequences=True),
    keras.layers.LSTM(100),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(vocab_size, activation='softmax')
])

# Train model with checkpoints
model_2.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
filepath = "./model_2_weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
history = model_2.fit(np.asarray(trainX),
         pd.get_dummies(np.asarray(trainy)),
         epochs = 300,
         batch_size = 128,
         verbose = 1)
