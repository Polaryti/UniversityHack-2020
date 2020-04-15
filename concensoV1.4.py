'''
En este modelo se ha realizado:
- CONSTRUCTIONYEAR -> Cambiar por antiguedad (NO CAMBIA NADA)
- MAXBUILDINGFLOOR -> Las entradas null de -1 a 0 (NO CAMBIA NADA)
- CADASTRALQUALITYID -> Transformar a one-hot
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
import random
from datasets_get import getX, getY, get_estimar_data, get_modelar_data, get_modelar_data_ids, reduce_geometry_average, reduce_colors

print('Comienzo')
X_modelar = reduce_geometry_average(getX(get_modelar_data_ids()))
X_modelar = reduce_colors(X_modelar)
print('Geo1')
X_estimar = reduce_geometry_average(get_estimar_data())
X_estimar = reduce_colors(X_estimar)
print('Geo2')


Y_modelar = np.array(getY(get_modelar_data()))
Y_modelar = Y_modelar[1:, :]
print(Y_modelar.shape)

X_estimar = np.array(X_estimar)
X_estimar = X_estimar[1:, :] # Quitamos el nombre de las variables
print(X_estimar.shape)

X_modelar = np.array(X_modelar)
X_modelar = X_modelar[1:, :] # Quitamos el nombre de las variables
print(X_modelar.shape)


# Variable que contendrá las muestras separadas por clase
data_per_class = []

# Añadimos una lista vacía por clase
for _ in range(7):         
    data_per_class.append([])
# Añadimos a la lista de cada clase las muestras de esta
for i in range(len(X_modelar) + 1):
    data_per_class[int(Y_modelar[i])].append(X_modelar[i] + [Y_modelar[i]])


# Variable que contendrá los datos procesados
data_proc = []

# Variable que contendrá las muestras a predecir
data_predict = []


# Variable que contendra las predicciones globales de cada muestra
predictions = {}

# Número de iteraciones total por módelo
iterations = 20

# Si True, muestra información de cada modelo local tras entrenarlo
debug_mode = True

# Variable en el rango (0.0 - 1.0) que indica el procentaje de muestras de validación
test_avg = 0.06

sum_avg = 0

for ite in range(iterations):
    data_proc = []
    # Muestras de la clase RESIDENTIAL
    random.shuffle(data_per_class[0])
    data_proc += data_per_class[0][:5250]

    # Muestras de las otras clases
    for i in range(6):
        data_proc += data_per_class[i + 1]
        
    # Volvemos a convertir los datos una vez procesados a una matriz
    data_proc = np.array(data_proc)

    # Obtenemos una separación del conjunto de train y test equilibrado (respecto a porcentaje de cada clase)
    pos = len(data_proc[0]) - 1
    X, Y = (data_proc[:, :pos], data_proc[:, pos])

    sss = StratifiedShuffleSplit(n_splits = 1, test_size = test_avg)

    for train_index, test_index in sss.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]


    # Mostramos el porcentaje de entrenamiento
    print('Entrenamiento completo al {}%'.format(ite/iterations * 100))
    
    # Modelo XGB
    model = xgb.XGBClassifier(
        # General parameters
        # Tree Booster parameters
        eta = 0.15,
        max_depth = 10,
        n_estimators = 240,
        tree_method = 'exact',
        # Learning task parameters
        objective = 'multi:softmax',
        num_class =  7,
        eval_metric = 'merror',
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if debug_mode:
        #print('Matriz de confusión:\n{}\n'.format(confusion_matrix(y_test, y_pred)))
        #print('Informe de clasificación:\n{}\n'.format(classification_report(y_test, y_pred)))
        print('XGBoost ({})'.format(ite))
        print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))
        print('Recall (macro): {}'.format(recall_score(y_test, y_pred, average = 'macro')))
        print('F1 (macro): {}'.format(f1_score(y_test, y_pred, average = 'macro')))
        sum_avg += f1_score(y_test, y_pred, average = 'macro')
    
    predictions_aux = model.predict(data_predict[:, 1:].astype('float32'))  
    for i in range(len(data_predict)):
        if (data_predict[i, 0] not in predictions):
            predictions[data_predict[i, 0]] = [int(predictions_aux[i])]
        else:
            predictions[data_predict[i, 0]].append(int(predictions_aux[i]))
        
    # Modelo RandomForest
    model = RandomForestClassifier(
        criterion = 'entropy',
        n_jobs = -1,
        max_features = None,
        n_estimators = 400,
        max_depth = 50,
        min_samples_split = 3,
        min_samples_leaf = 1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if debug_mode:
        # print('Matriz de confusión:\n{}\n'.format(confusion_matrix(y_test, y_pred)))
        # print('Informe de clasificación:\n{}\n'.format(classification_report(y_test, y_pred)))
        print('RF ({})'.format(ite))
        print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))
        print('Recall (macro): {}'.format(recall_score(y_test, y_pred, average = 'macro')))
        print('F1 (macro): {}'.format(f1_score(y_test, y_pred, average = 'macro')))
        sum_avg += f1_score(y_test, y_pred, average = 'macro')
    
    print('\n')
    predictions_aux = model.predict(X_estimar[:, 1:].astype('float32'))
    for i in range(len(X_estimar)):
        if (data_predict[i, 0] not in predictions):
            predictions[X_estimar[i, 0]] = [int(predictions_aux[i])]
        else:
            predictions[X_estimar[i, 0]].append(int(predictions_aux[i]))
print('Entrenamiento completo {}'.format(sum_avg / (iterations * 2)))

# Diccionario para decodificar el nombre de las clases
categorical_decoder_class = {0: 'RESIDENTIAL',
    1: 'INDUSTRIAL',
    2: 'PUBLIC',
    3: 'OFFICE',
    4: 'OTHER',
    5: 'RETAIL',
    6: 'AGRICULTURE'}

def most_frequent(lst): 
    return max(set(lst), key = lst.count) 

with open(r'Resultados/Minsait_Universitat Politècnica de València_Astralaria_FE.txt', 'w') as write_file:
    write_file.write('ID|CLASE\n')
    for sample in data_predict:
        write_file.write('{}|{}\n'.format(sample[0], categorical_decoder_class[most_frequent(predictions[sample[0]])]))