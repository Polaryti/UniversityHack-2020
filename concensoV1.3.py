'''
En este modelo se ha realizado:
- CONSTRUCTIONYEAR -> Cambiar por antiguedad (NO CAMBIA NADA)
- MAXBUILDINGFLOOR -> Las entradas null de -1 a 0 (NO CAMBIA NADA)
- CADASTRALQUALITYID -> Transformar a one-hot
'''
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
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


# Finalmente convertimos las muestras preprocesadas a una matriz
data = np.array(data).astype('float32')

# Variable que contendrá las muestras separadas por clase
data_per_class = []

# Añadimos una lista vacía por clase
for _ in range(7):         
    data_per_class.append([])
# Añadimos a la lista de cada clase las muestras de esta
for sample in data:
    data_per_class[int(sample[len(sample) - 1])].append(sample)



# Variable que contendrá los datos procesados
data_proc = []

# Variable que contendrá las muestras a predecir
data_predict = []

# Mismo procesamiento de datos que para el conjunto inicial
with open(r'Data\Estimar_UH2020.txt') as read_file:
    # La primera linea del documento es el nombre de las variables, no nos interesa
    read_file.readline()
    # Leemos línea por línea adaptando las muestras al formato deseado (codificar el valos catastral)
    for line in read_file.readlines():
        # Eliminamos el salto de línea final
        line = line.replace('\n', '')
        # Separamos por el elemento delimitador
        line = line.split('|')
        # Cambiamos CONTRUCTIONYEAR a la antiguedad del terreno
        line[52] = 2020 - int(line[52])
        if line[53] is '':
            line[53] = 0
        # Codificamos CADASTRALQUALITYID y arreglamos la muestra
        data_predict.append(line[:54] + categorical_encoder_catastral[line[54]])

# Finalmente convertimos las muestras preprocesadas a una matriz (no númerica, nos interesa el id esta vez)
data_predict = np.array(data_predict)

# Variable que contendra las predicciones globales de cada muestra
predictions = {}

# Número de iteraciones total por módelo
iterations = 10

# Variable anterior, inicializada de nuevo
predictions = {}

# Si True, muestra información de cada modelo local tras entrenarlo
debug_mode = True

# Variable en el rango (0.0 - 1.0) que indica el procentaje de muestras de validación
test_avg = 0.2

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
        n_estimators = 200,
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
    predictions_aux = model.predict(data_predict[:, 1:].astype('float32'))
    for i in range(len(data_predict)):
        if (data_predict[i, 0] not in predictions):
            predictions[data_predict[i, 0]] = [int(predictions_aux[i])]
        else:
            predictions[data_predict[i, 0]].append(int(predictions_aux[i]))
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

with open(r'Resultados/Minsait_Universitat Politècnica de València_Astralaria3.txt', 'w') as write_file:
    write_file.write('ID|CLASE\n')
    for sample in data_predict:
        write_file.write('{}|{}\n'.format(sample[0], categorical_decoder_class[most_frequent(predictions[sample[0]])]))