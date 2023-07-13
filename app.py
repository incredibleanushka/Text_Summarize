from flask import Flask,render_template,url_for,request

import pandas as pd 
import joblib
import numpy as np
data = 'tennis.csv'
from nltk.corpus import stopwords
import requests


app = Flask(__name__)
# files = UploadSet('files', ALL)
# app.config['UPLOADED_FILES_DEST'] = ''
# app.config['ALLOWED_EXTENSIONS'] = ['CSV']
# configure_uploads(app,files)

from nltk.tokenize import sent_tokenize
df= pd.read_csv(data)

#print(df)
sentences = []
for column in df['article_text']:
    sentences.append(sent_tokenize(column))

sentences = [y for x in sentences for y in x]
#print(sentences)
import nltk
nltk.download('stopwords')
word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
#print(f)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()

clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
clean_sentences = [s.lower() for s in clean_sentences]
stop_words = stopwords.words('english')
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new
clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
sentence_vectors = []
for i in clean_sentences:
  if len(i) != 0:
    v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
  else:
    v = np.zeros((100,))
  sentence_vectors.append(v)
  sim_mat = np.zeros([len(sentences), len(sentences)])
from sklearn.metrics.pairwise import cosine_similarity
for i in range(len(sentences)):
  for j in range(len(sentences)):
    if i != j:
      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
      #print(sim_mat)


import networkx as nx

nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)
joblib.dump(scores, 'text_sum.pkl')


@app.route("/")
def msg():
    return render_template('home.html')

#joblib.dump(scores, 'text_sum.pkl')
# text_sum= open('text_sum.pkl','rb')
# scores = joblib.load(text_sum)

@app.route("/summarize",methods =['POST','GET'])
def getSummary():
    
    

    body=request.form['data']
    print(body)
    text_sum= open('text_sum.pkl','rb')
    scores = joblib.load(text_sum)
    result = scores(body , num_sentences=6)
    return render_template('result.html',result=result)
# joblib.dump(scores, 'text_sum.pkl')
# text_sum= open('text_sum.pkl','rb')
# scores = joblib.load(text_sum)



if __name__ =="__main__":
    app.run(debug=True,port=8000)



