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

CLASS = 'CLASS'

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
        labels = read_file.readline().split('|')
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
                if line[54] in categorical_encoder_catastral:
                    line[54] = categorical_encoder_catastral[line[54]]
                data_list.append(line[1:])

    # Finalmente convertimos las muestras preprocesadas a una matriz
    data_list = np.array(data_list).astype('float32')
    # Convertimos dicha matriz a un dataframe de pandas
    #df = pd.DataFrame(data = data_list).rename(columns={np.int64(66):'CLASS'})
    df = pd.DataFrame(data=data_list).rename(columns={66:'CLASS'})
    return df


def get_mod_data_original():
    # Variable que contendrá las muestras
    data = []

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
            data.append(line[1:])

    # Finalmente convertimos las muestras preprocesadas a una matriz
    data = np.array(data).astype('float32')
    df = pd.DataFrame(data=data).rename(columns={54:'CLASS'})
    return df

def get_modelar_data_ids(missing_value = 0, one_hot = True):
    # Variable que contendrá las muestras
    data_list = []
    # with open(r'/home/asicoder/Documentos/Projects/Python/UniversityHack-2020/Data/Modelar_UH2020.txt') as read_file:
    with open(r'Data/Modelar_UH2020.txt') as read_file:
        # La primera linea del documento es el nombre de las variables, no nos interesa
        labels = read_file.readline().split('|')
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
                data_list.append(line[:54] + categorical_encoder_catastral_onehot[line[54]] + [line[55]])
            else:
                if line[54] in categorical_encoder_catastral:
                    line[54] = categorical_encoder_catastral[line[54]]
                data_list.append(line[1:])

    # Finalmente convertimos las muestras preprocesadas a una matriz
    a = np.array(data_list)
    ids = a[:, 0]
    a = np.delete(a, 0, axis=1)
    a = a.astype('float32')
    dfids = pd.DataFrame(data=ids)
    dfa = pd.DataFrame(data=a).rename(columns={66:'CLASS'})
    dfa[0] = dfids
    return dfa


def getX(modelar_df):
    return modelar_df.loc[:, modelar_df.columns!=CLASS]


def getY(modelar_df):
    return modelar_df.loc[:, modelar_df.columns == CLASS]


def reduce_dimension_modelar(modelar_df, num=30):
    if num > 55:
        print('num no mayor a 55')
    else:
        importance_df = pd.read_csv('Importancia de parametros.csv')
        indexes_list = list(importance_df['Index'])
        indexes_list[::-1]
        indexes_quited = []
        i = 0
        j = 0
        while j < num:
            if not 53 <= indexes_list[i] <= 65 and indexes_list[i] != 1:
                indexes_quited.append(indexes_list[i])
                del modelar_df[indexes_list[i]]
                j += 1
            i+=1
        return modelar_df


def reduce_colors(df):
    #Quita los deciles 2,3,4,6,7 y 8 de cada color
    indices_start = [4, 5, 6, 8, 9, 10]
    for i in range(len(indices_start)):
        df.drop([indices_start[i], indices_start[i]+11, indices_start[i]+22, indices_start[i]+33],inplace=True,axis=1)
    return df


def reduce_geometry_average(df):
    avgs = []
    for i in range(df.shape[0]):
        avgs.append((df.loc[i, 48] + df.loc[i, 49] + df.loc[i, 50] + df.loc[i, 51]) / 4)
    del df[48]
    del df[49]
    del df[50]
    del df[51]
    df['GEOM_AVG'] = avgs
    
    return df


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
    ids = data_predict[:, 0]
    data_predict = np.delete(data_predict, 0, axis=1)
    data_predict = data_predict.astype('float32')
    dfids = pd.DataFrame(data=ids)
    dfa = pd.DataFrame(data=data_predict)
    dfa[0] = dfids
    return dfa

def get_estimar_ids():
    res = []
    with open(r'Data/Estimar_UH2020.txt') as read_file:
        read_file.readline()
        for sample in read_file.readlines():
            print(sample)
            sample = sample.split('|')
            print(sample)
            res.append(sample[0])

    return res