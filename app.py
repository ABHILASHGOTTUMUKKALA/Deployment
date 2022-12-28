import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# create Flask app
Flask_app = Flask(__name__)
model = pickle.load(open("model_pickle", "rb"))

@Flask_app.route("/")
def Home():
    return render_template("index.html")

@Flask_app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    def Prediction():
        if value == 0:
            print('not Churn')
        else:
            print('Churn')
    prediction = model.predict(features)                
    return render_template("index.html", prediction_text = "The Member may {}".format(prediction))

if __name__ == "__main__":
    Flask_app.run(debug=True)