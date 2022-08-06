from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import pickle

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences

from flask import Flask, render_template, request, Response, flash


app = Flask(__name__)

MODEL_PATH = 'Models/best_model.h5'

model = load_model(MODEL_PATH)

with open('Models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

print('Model loaded. Check http://127.0.0.1:5000/')

def predict_class(text):
    '''Function to predict sentiment class of the passed text'''
    
    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    max_len=50
    
    # Transforms text to a sequence of integers using a tokenizer object
    xt = tokenizer.texts_to_sequences(text)
    # Pad sequences to the same length
    xt = pad_sequences(xt, padding='post', maxlen=max_len)
    # Do the prediction using the loaded model
    yt = model.predict(xt).argmax(axis=1)
    # Print the predicted sentiment
    print('The predicted sentiment is', sentiment_classes[yt[0]])
    return sentiment_classes[yt[0]]

@app.route('/')
def index():
	return render_template("index.html", data = "Hey")

@app.route('/SENTIMENTAL_CHECK',methods=["POST"])
def SENTIMENTAL_CHECK():
    text = request.form['TEXT']
    print(text)
    answer = predict_class([text])
    return render_template('prediction.html' ,sentence = answer)



if __name__ == "__main__":
	app.run(debug=True)
    