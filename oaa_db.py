 # coding=utf-8
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import recall_score,accuracy_score,confusion_matrix, f1_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import tests_modelar
from sampling import smote_tomek, smote_enn
import XGB_RandomForestBasic_bunyol

INDEX = 0
CLASS = 54

estimar_df = XGB_RandomForestBasic_bunyol.get_estimar_data()
modelar_df = XGB_RandomForestBasic_bunyol.get_modelar_data()

#Obtener train 80% y test 20% aleatoriamente.
X_modelar = modelar_df.loc[:, modelar_df.columns!=CLASS]
y_modelar = modelar_df.loc[:, CLASS]
X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(X_modelar,y_modelar,test_size = 0.2,shuffle=True)

#X_train_notrandom, X_test_notrandom, y_train_notrandom, y_test_notrandom = train_test_split(X_modelar, y_modelar, test_size=0.2, shuffle=True, stratify=)

#Obtener train 80% y test 20% con los porcentajes de cada clase definidos.
modelar_train_test_80_20 = tests_modelar.dividir_dataset(modelar_df)
modelar_train_80_20 = modelar_train_test_80_20[0]
modelar_test_80_20 = modelar_train_test_80_20[1]
X_train_80_20  = modelar_train_80_20.loc[:, modelar_train_80_20.columns != CLASS]
y_train_80_20  = modelar_train_80_20.loc[:, CLASS]
X_test_80_20   = modelar_test_80_20.loc[:, modelar_test_80_20.columns != CLASS]
y_test_80_20   = modelar_test_80_20.loc[:, CLASS]


#Primer clasificador, con el dataset sin balancear.
model = xgb.XGBClassifier()
#OAA 1º probamos con este
clf0 = OneVsRestClassifier(model,n_jobs=3).fit(X_modelar,y_modelar)
pred = clf0.predict(X_test_80_20)

scores = []

scores.append((f1_score(y_test_80_20,pred,average='macro'), precision_score(y_test_80_20,pred,average='micro'), recall_score(y_test_80_20,pred,average='micro'), accuracy_score(y_test_80_20,pred)))

results = pd.DataFrame(scores,columns=['f1','precision','recall','accuracy'])

print(results)
print(confusion_matrix(y_test_80_20,pred))

#Balanceo de datos
#SMOTE y ENN
X_s_enn, y_s_enn = smote_enn(X_train_80_20, y_train_80_20)
#SMOTE y Tomek
X_s_tomek, y_s_tomek = smote_tomek(X_train_80_20, y_train_80_20)
#SMOTE y CMTNN


#Entrenamiento 2º clasificador para cada par.
clfENN = OneVsRestClassifier(model, n_jobs=3).fit(X_s_enn, y_s_enn)
clfTomek = OneVsRestClassifier(model, n_jobs=3).fit(X_s_tomek, y_s_tomek)

predENN = clfENN.predict(X_test_80_20)
predTomek = clfTomek.predict(X_test_80_20)

#Obtención de resultados.
scoreENN = []
scoreTomek = []

scoreENN.append((f1_score(y_test_80_20,predENN,average='macro'), precision_score(y_test_80_20,predENN,average='micro'), recall_score(y_test_80_20,predENN,average='micro'), accuracy_score(y_test_80_20,predENN)))
scoreTomek.append((f1_score(y_test_80_20,predTomek,average='macro'), precision_score(y_test_80_20,predTomek,average='micro'), recall_score(y_test_80_20,predTomek,average='micro'), accuracy_score(y_test_80_20,predTomek)))

resultsENN = pd.DataFrame(scoreENN,columns=['f1','precision','recall','accuracy'])
resultsTomek = pd.DataFrame(scoreTomek,columns=['f1','precision','recall','accuracy'])

print(resultsENN)
print(resultsTomek)