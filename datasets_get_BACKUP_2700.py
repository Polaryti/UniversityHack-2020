 # coding=utf-8
import os.path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score
import random

scriptpath = os.path.dirname(__file__)

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
    
def get_modelar_data(missing_value = 0, one_hot = True):
    # Variable que contendrá las muestras
    data_list = []
    # with open(r'/home/asicoder/Documentos/Projects/Python/UniversityHack-2020/Data/Modelar_UH2020.txt') as read_file:
    with open(r'Data/Modelar_UH2020.txt') as read_file:
        # La primera linea del documento es el nombre de las variables, no nos interesa
        labels = np.array(read_file.readline())
        # Leemos línea por línea adaptando las muestras al formato deseado (codificar el valor catastral y la clase)
        for line in read_file.readlines():
            # Eliminamos el salto de línea final
            line = line.replace('\n', '')
            # Separamos por el elemento delimitador
            line = line.split('|')
            # Cambiamos CONTRUCTIONYEAR a la antiguedad del terreno
            line[52] = 2020 - int(line[52])
            if line[53] is '':
                line[53] = missing_value
            line[55] = categorical_encoder_class[line[55]]
            # Codificamos CADASTRALQUALITYID y arreglamos la muestra
            if one_hot:
                data_list.append(line[1:54] + categorical_encoder_catastral_onehot[line[54]] + [line[55]])
            else:
                data_list.append(line[1:])

    # Finalmente convertimos las muestras preprocesadas a una matriz
    data_list = np.array(data_list)
<<<<<<< HEAD
    #Convertimos dicha matriz a un dataframe de pandas.
    modelar_df = pd.DataFrame(data = data_list)
    print(modelar_df.shape)
    return modelar_df
=======
    # Convertimos dicha matriz a un dataframe de pandas
    return pd.DataFrame(data = data_list, columns = labels)
>>>>>>> 5cb10d47ce37cc91df7887205eb811e1127b4597


def get_estimar_data(missing_value = 0, one_hot = True):
    # Variable que contendrá las muestras a predecir
    data_predict = []
    # with open(r'/home/asicoder/Documentos/Projects/Python/UniversityHack-2020/Data/Estimar_UH2020.txt') as read_file:
    with open(r'Data/Estimar_UH2020.txt') as read_file:
        # La primera linea del documento es el nombre de las variables (al ser un Pandas Dataframe hay que añadirla)
        labels = np.array(read_file.readline())
        # Leemos línea por línea adaptando las muestras al formato deseado (codificar el valos catastral)
        for line in read_file.readlines():
            # Eliminamos el salto de línea final
            line = line.replace('\n', '')
            # Separamos por el elemento delimitador
            line = line.split('|')
            # Cambiamos CONTRUCTIONYEAR a la antiguedad del terreno
            line[52] = 2020 - int(line[52])
            if line[53] is '':
                line[53] = missing_value
            if one_hot:
                data_predict.append(line[:54] + categorical_encoder_catastral_onehot[line[54]])
            else:
                data_predict.append(line)

    # Finalmente convertimos las muestras preprocesadas a una matriz
    data_predict = np.array(data_predict)

    # Convertimos la matriz a un dataframe de pandas
    return pd.DataFrame(data = data_predict, columns = labels)