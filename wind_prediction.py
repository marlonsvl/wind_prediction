import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(4)
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dense, Conv1D, Flatten, Dropout
from keras.layers import Dense, LSTM
import keras.backend as K
import base64
from keras.layers import Activation, Dense, BatchNormalization, TimeDistributed
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
import math


@st.cache(persist=True)
def load_data():
    data = pd.read_csv(r'UGE_01_Filter.csv', index_col='DATE', parse_dates=['DATE'])
    return data

@st.cache(persist=True)
def split(df):
    train = df.loc[:'2017'].iloc[:,0:3]
    val = df.loc['2018':].iloc[:,0:3]
    # Normalización del set de entrenamiento
    sc = MinMaxScaler(feature_range=(0,1))
    scaled_train = sc.fit_transform(train)
    return val, scaled_train, sc

@st.cache(persist=True)
def train_data(scaled_train, time_step):
    #time_step = 24
    X_train = []
    Y_train = []
    m = len(scaled_train)

    for i in range(time_step,m):
        # X: bloques de "time_step" datos: 0-time_step, 1-time_step+1, 2-time_step+2, etc
        X_train.append(scaled_train[i-time_step:i,0])

        # Y: el siguiente dato
        Y_train.append(scaled_train[i,0])
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    #print(X_train.shape)
    # Reshape X_train para que se ajuste al modelo en Keras
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train, Y_train

# Accuracy
def soft_acc(y_true, y_pred):
	return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

#@st.cache(persist=True)
def get_history(X_train, Y_train, epochs):
	# Red LSTM
	dim_entrada = (X_train.shape[1],1)
	dim_salida = 1
	na = 24
	#epochs = 5
	# Model
	modelo = Sequential()
	modelo.add(LSTM(units=na, input_shape=dim_entrada))
	modelo.add(BatchNormalization())
	modelo.add(Dense(24))
	modelo.add(Activation('sigmoid'))
	modelo.add(Dense(12))
	modelo.add(Activation('sigmoid'))
	modelo.add(Dense(units=dim_salida))
	modelo.compile(optimizer='adam', loss='mse', metrics=[soft_acc,'mse', 'mae'])
	history = modelo.fit(X_train,Y_train,epochs=epochs,batch_size=24,validation_split=0.01, shuffle=True)
	return history, modelo

def plot_metrics(metrics_list):
    if 'Accuracy model' in metrics_list:
        st.subheader("Accuracy")
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        st.pyplot()

def plot_accuracy(history):
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	st.pyplot()

#@st.cache(persist=True)
def validation(val, sc, time_step):
	# Validación (predicción del valor de Average Power)
	#
	x_test = val.values
	x_test_g = x_test
	x_test = sc.transform(x_test)

	np.set_printoptions(suppress=True)

	X_test = []
	for i in range(time_step,len(x_test)):
	    X_test.append(x_test[i-time_step:i,0])
	X_test = np.array(X_test)
	X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
	######################
	X_test_G = []
	for i in range(time_step,len(x_test_g)):
	    X_test_G.append(x_test_g[i-time_step:i,0])
	X_test_G = np.array(X_test_G)
	X_test_G = np.reshape(X_test_G, (X_test_G.shape[0],X_test_G.shape[1],1))
	return X_test, X_test_G

def graficar_predicciones(real, prediccion):
    real=real[1000:1500]
    prediccion=prediccion[1000:1500]
    plt.figure(figsize=(16, 8))
    plt.plot(real[0:len(prediccion)],color='red', label='True Future')
    plt.plot(prediccion, color='blue', label='LSTM Prediction')
    plt.ylim(1.1 * np.min(prediccion)/2, 1.1 * np.max(prediccion))
    plt.xlabel('Date')
    plt.ylabel('Average Power')
    plt.legend()
    st.pyplot()

def MAE(predict,target):
    return (abs(predict-target)).mean()

def MSE(predict,target):
    return ((predict-target)**2).mean()

def RMSE(predict, target):
    return np.sqrt(((predict - target) ** 2).mean())

def MAPE(predict,target):
    return ( abs((target - predict) / target).mean()) * 100

def RMSLE(predict, target):
    total = 0 
    for k in range(len(predict)):
        LPred= np.log1p(predict[k]+1)
        LTarg = np.log1p(target[k] + 1)
        if not (math.isnan(LPred)) and  not (math.isnan(LTarg)): 
            total = total + ((LPred-LTarg) **2)
        
    total = total / len(predict)        
    return np.sqrt(total)

def R2(predict, target):
    return 1 - (MAE(predict,target) / MAE(target.mean(),target))
def R_SQR(predict, target):
    r2 = R2(predict,target)
    return np.sqrt(r2)

def R2_ADJ(predict, target, k):
    r2 = R2(predict,target)
    n = len(target)
    return (1 -  ( (1-r2) *  ( (n-1) / (n-(k+1)) ) ) )


def main():
    #time_step = 24
    st.title("Web App for Wind Power Forecasting for the Villonaco Wind Farm")
    st.sidebar.title("Model Hyperparameters")
    st.sidebar.markdown("💨")
    time_step = st.sidebar.number_input("Time step", 1, 30, step=1, key='time_step')
    epochs = st.sidebar.number_input("Epochs", 1, 300, step=10, key='epoch')
    
    df = load_data()
    
    val, scaled_train, sc = split(df)
    X_train, Y_train = train_data(scaled_train, time_step)

    #st.sidebar.subheader("Choose Model")
    #predictor = st.sidebar.selectbox("Predictor", ("LSTM", "Otro"))
    # desactivar los warnings de las imagenes
    st.set_option('deprecation.showPyplotGlobalUse', False)

    #st.sidebar.subheader("Model Hyperparameters")
    #choose parameters
    #C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
    #kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
    #gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')
    #aerogenerador = st.sidebar.selectbox("Wind Turbine", (1,2,3,4,5,6,7,8,9,10,11), key='aero')
    #metrics = st.sidebar.multiselect("What metrics to plot?", ('Accuracy model', 'Otro'))
    if st.sidebar.button("Train/Predict", key='train'):
    	st.header("LSTM Results")
    	history, modelo = get_history(X_train, Y_train, epochs)
    	plot_accuracy(history)
    	#if st.sidebar.button("Predict", key='predict'):
    	X_test, X_test_G = validation(val, sc, time_step)
    	prediccion = modelo.predict(X_test)
    	prediccion = sc.inverse_transform(prediccion)
    	st.header("Predictions")
    	graficar_predicciones(val.values,prediccion)
    	#data = (str(datum[0]) for datum in prediccion)
    	df = pd.DataFrame(prediccion)
    	#chart = st.line_chart(df)
    	csv_data = df.to_csv('LSTM_Pred_Data.csv', index = True)
    	st.dataframe(df)
    	st.header("How accurate a forecast system is?")
    	expectations = np.array(val.values[0:len(prediccion)])
    	predictions = np.array(prediccion)
    	st.subheader('MAE: ' + str(MAE(predictions,expectations)))
    	st.subheader('MSE: ' + str(MSE(predictions,expectations)) )
    	st.subheader('RMSE: ' + str(RMSE(predictions,expectations)))
    	st.subheader('MAPE: ' + str(MAPE(predictions,expectations)) )
    	st.subheader('RMSLE: ' + str(RMSLE(predictions,expectations)[0]))
    	st.subheader('R2 : ' + str(R2(predictions,expectations)))
    	st.subheader('R : ' + str(R_SQR(predictions,expectations)))
    	st.subheader("Mean Absolute Percent Error: " + str((np.mean(np.abs((expectations - predictions) / expectations.mean()))*100)))
    	st.subheader("Accuracy: " + str((sum(abs(expectations - predictions)/expectations))/len(expectations)))
    	#f'<a href="data:file/csv;base64,{base64}" download="LSTM_Pred_Data.csv">Download csv file</a>' 
    	#st.text(data)
    	#chart = st.line_chart(data)
if __name__ == '__main__':
    main()


