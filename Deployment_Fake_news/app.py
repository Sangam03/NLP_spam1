from flask import Flask,jsonify,request,render_template
import numpy as np
import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import remove_stopwords

import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.metrics import accuracy_score
#WARNING LIBRARYS
import warnings
warnings.filterwarnings("ignore")
import joblib

stemmer = PorterStemmer()
lemmitizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

app=Flask(__name__)

model=joblib.load("cv_model.pkl")
ML_Model = joblib.load("Spam_mail_model.pkl")

def text_cleaner(text):
    
    #Remove all character except A-Z and a-z
    sent = re.sub('[^a-zA-Z]',' ',text)
    
    #Convert into lower case
    sent = sent.lower()
    
    #Steamming
    sent = " ".join([stemmer.stem(word) for word in str(sent).split()])
    
    #Remove stopwords
    sent = " ".join([stemmer.stem(word) for word in str(sent).split()
                    if(word not in stop_words)])
    
    return sent

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict',methods=["POST"])
def predict():
	if request.method=='POST':

		#data from UI
		raw_news=request.form['email']

		# Code
		print("Before Cleanning:- ",raw_news)

		# Clearning the raw News
		cleaned_news = text_cleaner(raw_news)
		print("After Cleanning",cleaned_news)

		#Vectorize the clear_news
		X = model.transform([cleaned_news])

		pred = ML_Model.predict(X)[0]

		if(pred==1):
			result= "{} : - News is Fake".format(raw_news)
		else:
			result= "{} : - News is Real".format(raw_news)

		return jsonify({"Prediction": result})



if __name__=='__main__':
	app.run(debug=True)