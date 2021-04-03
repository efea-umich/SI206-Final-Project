from keras.layers import *
from keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np

num_words = 75000


def getHeadlines():

    # Open and sort dataset into Onion and real headline arrays
    with open('OnionOrNot.csv', 'r', encoding='utf-8') as hand:
        headlines = pd.read_csv(hand)
        o = headlines[headlines["label"] == 1]
        r = headlines[headlines["label"] == 0]
    return o["text"].to_numpy(), r["text"].to_numpy()



onion, real = getHeadlines()

x = np.concatenate([onion, real])
y = np.concatenate([np.ones(onion.shape), np.zeros(real.shape)])

# Assign the same random permutation to x (headlines) and y (labels)
p = np.random.permutation(len(x))
x = x[p]
y = y[p]


# Assign token to each word present in headlines
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'`’‘')
tokenizer.fit_on_texts(x)
trainX = tokenizer.texts_to_sequences(x)
indexLen = len(tokenizer.word_index)

model = Sequential([
    Embedding(indexLen + 1, 32),
    LSTM(256),
    Dropout(0.2),
    LSTM(256),
    Dropout(0.2),
    Dense(1024, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(trainX, y, epochs=10)
