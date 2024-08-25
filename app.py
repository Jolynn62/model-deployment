from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np
from datetime import datetime

app = Flask(__name__)

model = load_model('models/knn_pipeline')
cols = ['FISCAL_YR', 'FISCAL_MTH', 'DIV_NAME', 'MERCHANT', 'CAT_DESC', 'TRANS_DT', 'AMT']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    # Convert TRANS_DT to datetime
    if 'TRANS_DT' in data_unseen.columns:
        try:
            data_unseen['TRANS_DT'] = data_unseen['TRANS_DT'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d'))
        except ValueError as e:
            return f"Date conversion error: {e}"

    # processed_list = [x for x in data_unseen]
    # # Return the processed list
    # return str(data_unseen.columns)

    prediction = predict_model(model, data=data_unseen, round = 0)
    prediction = int(prediction.Label[0])
    return render_template('home.html',pred='Detected Anomaly as: {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)