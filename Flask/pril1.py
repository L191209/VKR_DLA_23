import flask
from flask import Flask, render_template, request
import pickle
import sklearn 
# импорт библиотек 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error 
from joblib import dump, load

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation, Dropout


app = flask.Flask(__name__, template_folder = 'templates')
@app.route('/', methods = ['POST', 'GET'])

@app.route('/index', methods = ['POST', 'GET'])
def main():
    if flask.request.method =='GET':
        return render_template('main.html')

    if flask.request.method == 'POST':
        std_scaler_l=load('scaler_df.joblin')  # загружаем масштабатор
        with open('model_best.pkl', 'rb') as f:
            loaded_model = pickle.load(f) # загружаем модель
        # получаем входные данные
        p2 = float(flask.request.form['p2'])
        p3 = float(flask.request.form['p3'])
        p4 = float(flask.request.form['p4'])
        p5 = float(flask.request.form['p5'])
        p6 = float(flask.request.form['p6'])
        p7 = float(flask.request.form['p7'])
        p10 = float(flask.request.form['p10'])
        p11 = float(flask.request.form['p11'])
        p12 = float(flask.request.form['p12'])
        p13 = float(flask.request.form['p13'])
        # 1, 8, 9 параметры укажем 0
        prm_df = np.array([0,p2,p3,p4,p5,p6,p7,0,0,p10,p11,p12,p13])
        # стандартизируем все 13:
        prm_std= std_scaler_l.transform([prm_df])
        # удаляем 1, 8, 9
        prm_std = np.delete(prm_std, [0, 7, 8])
        df = pd.DataFrame(data = [prm_std])
        # предсказание по модели
        y3_pred = loaded_model.predict(df)

        return render_template('main.html', result = y3_pred)
    
if __name__=='__main__':
    app.run()
