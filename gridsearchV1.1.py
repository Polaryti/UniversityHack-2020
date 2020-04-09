'''
En este modelo se ha realizado Grid Search de Random Forest con SOLO 6000 MUESTRAS DE RESIDENTIAL
'''
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score
import random

# Diccionario para codificar los nombres de las clases
categorical_encoder_class = {'RESIDENTIAL': 0,
    'INDUSTRIAL': 1,
    'PUBLIC': 2,
    'OFFICE': 3,
    'OTHER': 4,
    'RETAIL': 5,
    'AGRICULTURE': 6
}

# Diccionario para codificar la variable categorica CADASTRALQUALITYID a un vector one-hot
categorical_encoder_catastral = {
    'A': [1, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,0],
    'B': [0, 1, 0, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,0],
    'C': [0, 0, 1, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,0],
    '1': [0, 0, 0, 1, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,0],
    '2': [0, 0, 0, 0, 1, 0, 0, 0, 0 ,0 ,0 ,0 ,0],
    '3': [0, 0, 0, 0, 0, 1, 0, 0, 0 ,0 ,0 ,0 ,0],
    '4': [0, 0, 0, 0, 0, 0, 1, 0, 0 ,0 ,0 ,0 ,0],
    '5': [0, 0, 0, 0, 0, 0, 0, 1, 0 ,0 ,0 ,0 ,0],
    '6': [0, 0, 0, 0, 0, 0, 0, 0, 1 ,0 ,0 ,0 ,0],
    '7': [0, 0, 0, 0, 0, 0, 0, 0, 0 ,1 ,0 ,0 ,0],
    '8': [0, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,1 ,0 ,0],
    '9': [0, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0 ,1 ,0],
    '""': [0, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,1]
}

# Variable que contendrá las muestras
data = []

with open(r'Data\Modelar_UH2020.txt') as read_file:
    # La primera linea del documento es el nombre de las variables, no nos interesa
    read_file.readline()
    # Leemos línea por línea adaptando las muestras al formato deseado (codificar el valor catastral y la clase)
    for line in read_file.readlines():
        # Eliminamos el salto de línea final
        line = line.replace('\n', '')
        # Separamos por el elemento delimitador
        line = line.split('|')
        # Cambiamos CONTRUCTIONYEAR a la antiguedad del terreno
        line[52] = 2020 - int(line[52])
        if line[53] is '':
            line[53] = 0
        line[55] = categorical_encoder_class[line[55]]
        # Codificamos CADASTRALQUALITYID y arreglamos la muestra
        data.append(line[1:54] + categorical_encoder_catastral[line[54]] + [line[55]])

random.shuffle(data)
data = np.array(data).astype('float32')
for _ in range(10):
    # Variable que contendrá las muestras separadas por clase
    data_per_class = []
    data_proc = []

    # Añadimos una lista vacía por clase
    for _ in range(7):         
        data_per_class.append([])
    # Añadimos a la lista de cada clase las muestras de esta
    for sample in data:
        data_per_class[int(sample[len(sample) - 1])].append(sample)
    # Muestras de la clase RESIDENTIAL
    random.shuffle(data_per_class[0])
    data_proc += data_per_class[0][:5000]

    # Muestras de las otras clases
    for i in range(6):
        data_proc += data_per_class[i + 1]
            
    # Volvemos a convertir los datos una vez procesados a una matriz
    data_proc = np.array(data_proc)

    np.random.shuffle(data_proc)

    param_dict = {
        'n_estimators': [750, 775, 800],
        'criterion': ['gini'],
        'min_samples_split': [4, 5, 6, 7, 8],
        'min_samples_leaf': [3, 4],
        'n_jobs': [-1]
    }

    gs = GridSearchCV(
        estimator = RandomForestClassifier(), 
        param_grid = param_dict,
        scoring = 'balanced_accuracy',
        n_jobs = -1
        )

    pred_pos = len(data_proc[0]) - 1
    gs.fit(data_proc[:,:pred_pos], data_proc[:,pred_pos])
    print(gs.best_score_)
    print(gs.best_params_)
