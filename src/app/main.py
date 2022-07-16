from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
from googletrans import Translator
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle 
import os

colunas=['tamanho','ano','garagem']
modelo=pickle.load(open('../../models/modelo.sav','rb'))

app = Flask(__name__)
translator = Translator()

app.config['BASIC_AUTH_USERNAME']=os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD']=os.environ.get('BASIC_AUTH_PASSWORD')

basic_auth= BasicAuth(app)

@app.route('/')
def home():
    return "sou um Rack russo"

@app.route('/sentimento/<frase>')
def sentimento(frase):
    frase_en = translator.translate(frase, dest='en')
    tb_en = TextBlob(frase_en.text)
    polaridade = tb_en.sentiment.polarity
    return "polaridade: {}".format(polaridade)

@app.route('/cotacao/', methods=['POST'])
@basic_auth.required
def cotacao():
    dados=request.get_json()
    dados_input=[dados[col] for col in colunas]
    preco=modelo.predict([dados_input])
    return jsonify(preco=preco[0])

app.run(debug=True, host='0.0.0.0')
