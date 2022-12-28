from flask import Flask,jsonify,request,render_template
import numpy as np
import joblib

app=Flask(__name__)

model=joblib.load("KNN _Iris_Model.pkl")

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict',methods=["POST"])
def predict():
	if request.method=='POST':

		#data from UI
		sl=request.form['sepal_length']
		sw=request.form['sepal_width']
		pl=request.form['petal_length']
		pw=request.form['petal_width']

		# Check valu comming or not
		print("sl:-",sl)
		print("sw:-",sw)
		print("pl:-",pl)
		print("pw:-",pw)

		# Type cast
		sl=float(sl)
		sw=float(sw)
		pl=float(pl)
		pw=float(pw)

		result=model.predict([[sl,sw,pl,pw]])[0]

		return jsonify({"Prediction": result})



if __name__=='__main__':
	app.run(debug=True)