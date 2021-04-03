from keras.layers import *
from keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import json
import pickle
import keras

def get_sarcasm_dataset():
    with open('./sarcasm_dataset.json', 'r', encoding='utf-8') as f:
        j = json.JSONDecoder().decode(f.read())
        headlines, labels = [line['headline'] for line in j], [line['is_sarcastic'] for line in j]
    return headlines, labels

def get_onion_dataset():
    with open('OnionOrNot.csv', 'r', encoding='utf-8') as hand:
        headlines = pd.read_csv(hand)
        o = headlines[headlines["label"] == 1]
        r = headlines[headlines["label"] == 0]
    return np.concatenate([o['text'].to_numpy(), r['text'].to_numpy()]), np.concatenate([np.ones(o['text'].to_numpy().shape), np.zeros(r['text'].to_numpy().shape)])

with open('./onion_tokenizer.pyc', 'rb') as pickleHand:
    tokenizer = pickle.load(pickleHand)
assert isinstance(tokenizer, Tokenizer)

headlines, labels = get_onion_dataset()
p = np.random.permutation(len(headlines))
headlines = headlines[p]
labels = labels[p]

max_len = 200
headlines = headlines
labels = labels
seqs = tokenizer.texts_to_sequences(headlines)
seqs = pad_sequences(seqs, max_len)
model = keras.models.load_model('./onion_harvester_woah.h5')
assert isinstance(model, keras.models.Model)
model.evaluate(np.asarray(seqs), np.asarray(labels))