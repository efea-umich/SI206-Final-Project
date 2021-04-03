import bs4
import requests
import json
import pandas as pd
import numpy as np
import csv

API_KEY = 'fc0b3586ef3d42b8bc475d2c9735efe8'
news_url = 'https://newsapi.org/v2/everything?sources=abc-news,bbc-news,cnn,fox-news,msnbc,nbc-news,politico,reuters&lang=en&pageSize=100&apiKey=fc0b3586ef3d42b8bc475d2c9735efe8'

def apiToFile():
    articles = []
    for i in range(1, 2):
        j = json.JSONDecoder().decode(requests.get(news_url + f'&page={i}').text)
        articles.append([item['title'] for item in j['articles']])
    with open('news_api.csv', 'w') as out:
        w = csv.writer(out)
        w.writerows(articles)




def getHeadlines():

    # Open and sort dataset into Onion and real headline arrays
    with open('OnionOrNot.csv', 'r', encoding='utf-8') as hand:
        headlines = pd.read_csv(hand)
        o = headlines[headlines["label"] == 1]
        #r = headlines[headlines["label"] == 0]
    with open('sarcasm_dataset.json', 'r') as f:
        j = json.JSONDecoder().decode(f.read())
        r = [line['headline'] for line in j if line['is_sarcastic'] == 0]
    #with open('10000_headlines.csv', 'r', encoding='utf-8') as hand:
        #r = pd.read_csv(hand)]
    return o["text"].to_numpy(), np.asarray(r)

