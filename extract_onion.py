import pandas
import numpy as np

with open("OnionOrNot.csv", 'r') as hand:
    df = pandas.read_csv(hand)
    with open("OnionYes.csv", 'w') as onion_np:
        df[df['label'] == 1].to_csv(onion_np)
