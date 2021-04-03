from keras.layers import *
from keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import pickle
import keras

with open('./onion_tokenizer.pyc', 'rb') as pickleHand:
    tokenizer = pickle.load(pickleHand)
assert isinstance(tokenizer, Tokenizer)

max_len = 200
seqs = tokenizer.texts_to_sequences([input()])
seqs = pad_sequences(seqs, max_len)
model = keras.models.load_model('./onion_harvester_woah.h5')
assert isinstance(model, keras.models.Model)
print(model.predict(seqs))