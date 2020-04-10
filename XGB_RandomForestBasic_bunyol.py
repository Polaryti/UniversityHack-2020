 # coding=utf-8
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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

#Lista de las categorías
categories_list = ['RESIDENTIAL', 'INDUSTRIAL', 'PUBLIC', 'OFFICE', 'OTHER', 'RETAIL', 'AGRICULTURE']

# Diccionario para codificar las variables no numéricas
categorical_encoder_catastral = {'A': -10,
    'B': -20,
    'C': -30,
    '""': 50
}

#Número de variables
VARS_NUM = 55


def get_modelar_data():
    # Variable que contendrá las muestras
    data_list = []
    #Diccionario con los nombres de las variables, útil para las visualizaciones posteriores.
    vars_dict = {}
    vars_list = []
    with open(r'Data/Modelar_UH2020.txt') as read_file:
        # La primera linea del documento es el nombre de las variables, no nos interesa
        read_file.readline()
        # Leemos línea por línea adaptando las muestras al formato deseado (codificar el valor catastral y la clase)
        for line in read_file.readlines():
            # Eliminamos el salto de línea final
            line = line.replace('\n', '')
            # Separamos por el elemento delimitador
            line = line.split('|')
            if line[54] in categorical_encoder_catastral:
                line[54] = categorical_encoder_catastral[line[54]]
                if line[54] is 50:
                    line[53] = -1
            line[55] = categorical_encoder_class[line[55]]
            # No nos interesa el identificador de la muestra, lo descartamos
            data_list.append(line[1:])
    # Finalmente convertimos las muestras preprocesadas a una matriz
    data_list = np.array(data_list).astype('float32')
    #Convertimos dicha matriz a un dataframe de pandas.
    modelar_df = pd.DataFrame(data = data_list)
    return modelar_df


def get_estimar_data():
    # Variable que contendrá las muestras a predecir
    data_predict = []

    # Mismo procesamiento de datos que para el conjunto inicial
    with open(r'Data/Estimar_UH2020.txt') as read_file:
        # La primera línea del documento es el nombre de las variables, no nos interesa
        read_file.readline()
        # Leemos línea por línea adaptando las muestras al formato deseado (codificar el valor catastral)
        for line in read_file.readlines():
            line = line.replace('\n', '')
            line = line.split('|')
            if line[54] in categorical_encoder_catastral:
                line[54] = categorical_encoder_catastral[line[54]]
                if line[54] is 50:
                    line[53] = -1
            data_predict.append(line)

    # Finalmente convertimos las muestras preprocesadas a una matriz (no numérica, nos interesa el id esta vez)
    data_predict = np.array(data_predict)

    #Convertimos dicha matriz a un dataframe de pandas.
    estimar_df = pd.DataFrame(data = data_predict)
    return estimar_df

def get_categories_list():
    return categories_list