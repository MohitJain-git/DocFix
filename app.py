import numpy as np
from flask import Flask, request, send_file, jsonify, render_template
import pickle
import pandas as pd
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from transformers import pipeline
from datetime import datetime

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/summarize')
def toSummarize():
    return render_template('summarize.html')

@app.route('/getkw')
def getKW():
    return render_template('keyword.html')

@app.route('/getkw', methods=['POST'])
def keywordfinder():
    text_input = request.form['text']
    model = pickle.load(open('keyfinder_model', 'rb'))
    features = model.get_feature_names()
    output = get_keywords(model, features, text_input)
    return render_template('keyword.html', keywords = output , input='{}'.format(text_input))

def sort_coo(coo_matrix):
    """Sort a dict with highest score"""
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature, score
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

def get_keywords(vectorizer, feature_names, doc):
    """Return top k keywords from a doc using TF-IDF method"""

    #generate tf-idf for the given document
    tf_idf_vector = vectorizer.transform([doc])
    
    #sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tf_idf_vector.tocoo())

    #extract only TOP_K_KEYWORDS
    keywords=extract_topn_from_vector(feature_names,sorted_items,10)
    
    return list(keywords.keys())




@app.route('/summarize', methods=['POST'])
def summarize():
    text_input = request.form['text']
    model = pickle.load(open('bert_model.pkl', 'rb'))
    tgt_text = model(text_input)
    word_count = len(tgt_text[0]['summary_text'].split())

    return render_template('summarize.html', summarized_text=tgt_text[0]['summary_text'], input='{}'.format(text_input), words=word_count)


@app.route('/sentiment')
def toSentiment():
    return render_template('sentiment.html')


@app.route('/sentiment', methods=['POST'])
def sentiment():
    text_input = request.form['text']
    sentimentmodel = pickle.load(open('new_sentiment.pkl', 'rb'))
    tgt_text = sentimentmodel(text_input)
    sentscore = tgt_text[0]["score"] * 100
    sentlabel = tgt_text[0]["label"]
    return render_template('sentiment.html', sentiment_label='This output tells you whether the text is +ve or -ve after sentiment analysis : {}'.format(sentlabel), sentiment_score='This no. denotes the surety of the model for the first output : \n {:.2f} %'.format(sentscore), input='{}'.format(text_input))


@app.route('/getdata')
def getData():
    return render_template('content.html')


@app.route('/getdata', methods=['POST'])
def download():
    query = request.form['query']
    url = 'https://news.google.com/search?q=' + query
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    a = soup.find('a', class_='DY5T1d RZIKme')
    print("Title: "+a.text)
    title = a.text
    href = a['href']
    article = Article('https://news.google.com/'+href)
    article.download()
    article.parse()
    data = article.text
    now = datetime.now()
    str = now.strftime("%m%d%Y_%H%M%S")
    # str = str.replace("/", "")
    # str = str.replace(',', '')
    path = 'docfix'+str+".txt"
    with open(path, "w") as text_file:
        text_file.write("Title : %s" % title + "\n")
        text_file.write("Data : %s" % data)

    return render_template('content.html', path=path, data=data, title=title, input='{}'.format(query))


@app.route('/download_file<file_path>')
def download_file(file_path=None):
    path = file_path
    return send_file(path, as_attachment=True)



if __name__ == "__main__":
    app.run(debug=True)
