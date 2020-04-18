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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import random
from datasets_get import getX, getY, get_estimar_data, get_modelar_data, get_modelar_data_ids, reduce_geometry_average, reduce_colors
from not_random_test_generator import dividir_dataset, random_undersample_residential
from feature_engineering import coordinates_fe, density_RGB_scale

print("Start")
#X_modelar = reduce_geometry_average(getX(get_modelar_data()))
#X_modelar = reduce_colors(X_modelar)

#X_estimar = reduce_geometry_average(get_estimar_data())
#X_estimar = reduce_colors(X_estimar)

modelar_df = random_undersample_residential(get_modelar_data_ids())
X_modelar = getX(modelar_df)
X_estimar = get_estimar_data()
Y_modelar = getY(modelar_df)

X_modelar, X_estimar, est_IDS = coordinates_fe(X_modelar, Y_modelar, X_estimar)

X_modelar = density_RGB_scale(X_modelar)
X_estimar = density_RGB_scale(X_estimar)

Y_modelar = Y_modelar.values

Y_modelar = Y_modelar[1:, :]
X_estimar = X_estimar[1:, :]
X_modelar = X_modelar[1:, :]

print("Dataframes charged")
# Variable que contendrá las muestras separadas por clase
data_per_class = []

# Añadimos una lista vacía por clase
for _ in range(7):         
    data_per_class.append([])

# Añadimos a la lista de cada clase las muestras de esta
for i in range(len(X_modelar)):
    data_per_class[int(Y_modelar[i])].append(X_modelar[i, 1:].tolist() + Y_modelar[i].tolist())


# Variable que contendrá los datos procesados
data_proc = []

# Variable que contendra las predicciones globales de cada muestra
predictions = {}

# Número de iteraciones total por módelo
iterations = 15

# Si True, muestra información de cada modelo local tras entrenarlo
debug_mode = True

# Variable en el rango (0.0 - 1.0) que indica el procentaje de muestras de validación
test_avg = 0.2

accuracy_avg = 0
precision_avg = 0
recall_avg = 0
f1_avg = 0

print('Start iterations\n')
for ite in range(iterations):
    data_proc = []
    # Muestras de la clase RESIDENTIAL
    random.shuffle(data_per_class[0])
    data_proc += data_per_class[0][:6000]

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
    print('Entrenamiento completo al {}%\n'.format(ite/iterations * 100))
    
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
        print('Precision (macro): {}'.format(precision_score(y_test, y_pred, average = 'macro')))
        print('Recall (macro): {}'.format(recall_score(y_test, y_pred, average = 'macro')))
        print('F1 (macro): {}'.format(f1_score(y_test, y_pred, average = 'macro')))
        accuracy_avg += accuracy_score(y_test, y_pred)
        precision_avg += precision_score(y_test, y_pred, average = 'macro')
        recall_avg += recall_score(y_test, y_pred, average = 'macro')
        f1_avg += f1_score(y_test, y_pred, average = 'macro')
    
    predictions_aux = model.predict(X_estimar[:, 1:].astype('float32'))  
    for i in range(len(X_estimar)):
        if (X_estimar[i, 0] not in predictions):
            predictions[X_estimar[i, 0]] = [int(predictions_aux[i])]
        else:
            predictions[X_estimar[i, 0]].append(int(predictions_aux[i]))
        
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
        print('Precision (macro): {}'.format(precision_score(y_test, y_pred, average = 'macro')))
        print('Recall (macro): {}'.format(recall_score(y_test, y_pred, average = 'macro')))
        print('F1 (macro): {}\n'.format(f1_score(y_test, y_pred, average = 'macro')))
        accuracy_avg += accuracy_score(y_test, y_pred)
        precision_avg += precision_score(y_test, y_pred, average = 'macro')
        recall_avg += recall_score(y_test, y_pred, average = 'macro')
        f1_avg += f1_score(y_test, y_pred, average = 'macro')
    
    print('\n')
    predictions_aux = model.predict(X_estimar[:, 1:].astype('float32'))
    for i in range(len(X_estimar)):
        if (X_estimar[i, 0] not in predictions):
            predictions[X_estimar[i, 0]] = [int(predictions_aux[i])]
        else:
            predictions[X_estimar[i, 0]].append(int(predictions_aux[i])) 

print('\nEntrenamiento completo\n')
print('Accuracy: {}'.format(accuracy_avg / (iterations * 2)))
print('Precision: {}'.format(precision_avg / (iterations * 2)))
print('Recall: {}'.format(recall_avg / (iterations * 2)))
print('F1: {}'.format(f1_avg / (iterations * 2)))

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

with open(r'Resultados/Res_FE-03', 'w') as write_file:
    write_file.write('ID|CLASE\n')
    for sample in X_estimar:
        write_file.write('{}|{}\n'.format(sample[0], categorical_decoder_class[most_frequent(predictions[sample[0]])]))