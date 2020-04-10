import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score
import random

#Lista de las categorías
categories_list = ['RESIDENTIAL', 'INDUSTRIAL', 'PUBLIC', 'OFFICE', 'OTHER', 'RETAIL', 'AGRICULTURE']

# Diccionario para codificar los nombres de las clases
categorical_encoder_class = {'RESIDENTIAL': 0,
    'INDUSTRIAL': 1,
    'PUBLIC': 2,
    'OFFICE': 3,
    'OTHER': 4,
    'RETAIL': 5,
    'AGRICULTURE': 6
}

# Diccionario para codificar las variables no numéricas
categorical_encoder_catastral = {'A': -10,
    'B': -20,
    'C': -30,
    '""': 50
}

# Diccionario para decodificar el nombre de las clases
categorical_decoder_class = {0: 'RESIDENTIAL',
    1: 'INDUSTRIAL',
    2: 'PUBLIC',
    3: 'OFFICE',
    4: 'OTHER',
    5: 'RETAIL',
    6: 'AGRICULTURE'}

# Diccionario para codificar la variable categorica CADASTRALQUALITYID a un vector one-hot
categorical_encoder_catastral_onehot = {
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

def get_categorical_encoder_class():
    return categorical_encoder_class

def get_categorical_encoder_catastral():
    return categorical_encoder_catastral

def get_categorical_decoder_class():
    return categorical_decoder_class

def get_categories_list():
    return categories_list

def get_categorical_encoder_catastral_onehot():
    return categorical_encoder_catastral_onehot
    
def get_modelar_data(os, missing_value=0, one_hot=True):
    # Variable que contendrá las muestras
    data_list = []
    path = ''
    if os == 'linux':
        path += '../Data/Modelar_UH2020.txt'
    else:
        path += '..\Data\Modelar_UH2020.txt'
    with open(path) as read_file:
        # La primera linea del documento es el nombre de las variables, no nos interesa
        data_list.append(read_file.readline())
        # Leemos línea por línea adaptando las muestras al formato deseado (codificar el valor catastral y la clase)
        for line in read_file.readlines():
            # Eliminamos el salto de línea final
            line = line.replace('\n', '')
            # Separamos por el elemento delimitador
            line = line.split('|')
            if line[54] in categorical_encoder_catastral:
                line[54] = categorical_encoder_catastral[line[54]]
                if line[54] is 50:
                    line[53] = missing_value
            line[55] = categorical_encoder_class[line[55]]
            # No nos interesa el identificador de la muestra, lo descartamos
            if one_hot:
                data_list.append(line[1:] + categorical_encoder_catastral[line[54]] + [line[55]])
            else:
                data_list.append(line[1:])
    # Finalmente convertimos las muestras preprocesadas a una matriz
    data_list = np.array(data_list)
    #Convertimos dicha matriz a un dataframe de pandas.
    modelar_df = pd.DataFrame(data = data_list)
    return modelar_df

def get_estimar_data(os, missing_value=0, one_hot=True):
    # Variable que contendrá las muestras a predecir
    data_predict = []
    path = ''
    if os == 'linux':
        path += '../Data/Modelar_UH2020.txt'
    else:
        path += '..\Data\Modelar_UH2020.txt'
    # Mismo procesamiento de datos que para el conjunto inicial
    with open(path) as read_file:
        # La primera línea del documento es el nombre de las variables, no nos interesa
        read_file.readline()
        # Leemos línea por línea adaptando las muestras al formato deseado (codificar el valor catastral)
        for line in read_file.readlines():
            line = line.replace('\n', '')
            line = line.split('|')
            if line[54] in categorical_encoder_catastral:
                line[54] = categorical_encoder_catastral[line[54]]
                if line[54] is 50:
                    line[53] = missing_value
            if one_hot:
                data_predict.append(line[:54] + categorical_encoder_catastral[line[54]])
            else:
                data_predict.append(line)

    # Finalmente convertimos las muestras preprocesadas a una matriz (no numérica, nos interesa el id esta vez)
    data_predict = np.array(data_predict)

    #Convertimos dicha matriz a un dataframe de pandas.
    estimar_df = pd.DataFrame(data = data_predict)
    return estimar_df