 # coding=utf-8
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import recall_score,accuracy_score,confusion_matrix, f1_score, precision_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from datasets_get import get_modelar_data
from not_random_test_generator import dividir_dataset
from sampling import smote_enn, smote_tomek
import random
import math
import datetime

INDEX = 0
CLASS = 66

modelar_df = get_modelar_data()
print(modelar_df.shape)
#Obtener train 80% y test 20% aleatoriamente.
X_modelar = modelar_df.loc[:, modelar_df.columns!=CLASS]
y_modelar = modelar_df.loc[:, CLASS]
X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(X_modelar,y_modelar,test_size = 0.2,shuffle=True)
print('1')
#Obtener train 80% y test 20% NO aleatoriamente.
modelar_train_test_80_20 = dividir_dataset(modelar_df)
modelar_train_80_20 = modelar_train_test_80_20[0]
modelar_test_80_20 = modelar_train_test_80_20[1]
print('2')
X_train_80_20  = modelar_train_80_20.loc[:, modelar_train_80_20.columns != CLASS]
print('3')
y_train_80_20  = modelar_train_80_20.loc[:, CLASS]
print('4')
X_test_80_20   = modelar_test_80_20.loc[:, modelar_test_80_20.columns != CLASS]
print('5')
y_test_80_20   = modelar_test_80_20.loc[:, CLASS]
print('6')

#Los datos ya están preprocesados con one-hot vectors.
#Modelo XGBClassifier
xgbClass = xgb.XGBClassifier()

def classifier1(f):
    print('Entra en Clasificador 1')
    #OAA 1º probamos con este
    ovsr1 = OneVsRestClassifier(xgbClass,n_jobs=3).fit(X_modelar,y_modelar)
    prediction = ovsr1.predict(X_test_80_20)
    print('Sale de clasificador 1')

    scores = []
    scores.append((f1_score(y_test_80_20,prediction,average='macro'), precision_score(y_test_80_20,prediction,average='micro'), recall_score(y_test_80_20,prediction,average='micro'), accuracy_score(y_test_80_20,prediction)))
    results = pd.DataFrame(scores,columns=['f1','precision','recall','accuracy'])

    f.write('-PRIMER CLASIFICADOR-\n')
    f.write(results)
    f.write('\n\n')
    return ovsr1, prediction

def data_balancing(method, f):
    #method: 0 SMOTE y ENN, 1 SMOTE y Tomek, 2 SMOTE Y CMTNN
    print('Entra en data balancing')
    f.write('-BALANCEO DE DATOS, técnica: ')
    if method == 0:
        #SMOTE y ENN
        f.write('SMOTE + ENN-\n')
        X_s_enn, y_s_enn = smote_enn(X_train_80_20, y_train_80_20)
        f.write(str('Número de componentes X: ' + X_s_enn.shape[0] + '\n'))
        f.write(str('Número de componentes y: ' + y_s_enn.shape[0] + '\n'))
        print('Sale de data balancing')
        return X_s_enn, y_s_enn
    else:
        #SMOTE y Tomek
        f.write('SMOTE + Tomek Links')
        X_s_tomek, y_s_tomek = smote_tomek(X_train_80_20, y_train_80_20)
        f.write(str('Número de componentes X: ' + X_s_tomek.shape[0] + '\n'))
        f.write(str('Número de componentes y: ' + y_s_tomek.shape[0] + '\n'))
        print('Sale de data balancing')
        return X_s_tomek, y_s_tomek


def classifier2(f, X_balanced, y_balanced):
    #Entrenamiento 2º clasificador para cada técnica balanceo de datos.
    print('entra 2º clasificador')
    ovsr2 = OneVsRestClassifier(xgbClass, n_jobs=4).fit(X_balanced, y_balanced)
    print('sale 2º clasificador')
    prediction = ovsr2.predict(X_test_80_20)

    scores = []
    scores.append((f1_score(y_test_80_20,prediction,average='macro'), precision_score(y_test_80_20,prediction,average='micro'), recall_score(y_test_80_20,prediction,average='micro'), accuracy_score(y_test_80_20,prediction)))
    results = pd.DataFrame(scores,columns=['f1','precision','recall','accuracy'])

    f.write('-SEGUNDO CLASIFICADOR-\n')
    f.write(results)
    f.write('\n\n')
    return ovsr2, prediction
    

#Paso final del algoritmo, decisión con thresolds.
#Empleamos las predicciones del primer clasificador (prediction)
#y del 2º con cada técnica (predENN, predTomek)
def get_pred_final(f, ovsr1, ovsr2, pred1, pred2, thresold1, thresold2):
    f.write('PREDICCIÓN FINAL PARA THRESOLD 1: {} Y THRESOLD 2: {}\n'.format(thresold1, thresold2))
    obs = X_test_80_20.shape[0]
    probClasses = ovsr1.classes_
    res = []
    for i in range(obs):
        arr1 = []
        for j in range(7):
            if pred1[i, j] > thresold1:
                arr1.append(probClasses[j])
        #Si solo hay una clase en la lista, es la que predecimos.
        if len(arr1) == 1:
            res.append(arr1[0])
        else:
            #Creamos 2ª lista para comprobar el 2º clasificador.
            arr2 = []
            for j in range(7):
                if pred2[i,j] > thresold2:
                    arr2.append(probClasses[j])
            #Repetimos, si solo hay una clase en la lista, es la que predecimos.
            if len(arr2) == 1:
                res.append(arr2[0])
            #Paso final, no hay más clasificadores.
            #Cogemos clase que ha obtenido mayor probabilidad EN EL 1º.
            #Es posible el empate?
            else:
                res.append(probClasses[pred1.index(min(pred1[i, :]))])
    return res


def informe():
    #Creamos text file para el informe con el día y la hora 
    filename = str('Resultados/OAA-DB' + str(datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")))
    f = open(filename, 'w+')
    ovsr1, pred1 = classifier1(f)
    for i in range(2):
        X_balanced, y_balanced = data_balancing(i, f)
        ovsr2, pred2 = classifier2(f, X_balanced, y_balanced)
        res = get_pred_final(f, ovsr1, ovsr2, pred1, pred2, 50, 50)
        scores = []
        scores.append((f1_score(y_test_80_20,res,average='macro'), precision_score(y_test_80_20,res,average='micro'), recall_score(y_test_80_20,res,average='micro'), accuracy_score(y_test_80_20,res)))
        results = pd.DataFrame(scores,columns=['f1','precision','recall','accuracy'])
        f.write(results)
        f.write('----------------------------------------------------------------------------------------\n\n')

informe()
