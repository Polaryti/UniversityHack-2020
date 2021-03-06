# Modelo utilizado en la fase I

import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
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

# Diccionario para codificar las variables no numéricas
categorical_encoder_catastral = {'A': -10,
    'B': -20,
    'C': -30,
    '""': 50
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
        if line[54] in categorical_encoder_catastral:
            line[54] = categorical_encoder_catastral[line[54]]
            if line[54] is 50:
                line[53] = -1
        line[55] = categorical_encoder_class[line[55]]
        # No nos interesa el identificador de la muestra, lo descartamos
        data.append(line[1:])

# Finalmente convertimos las muestras preprocesadas a una matriz
data = np.array(data).astype('float32')

# Variable que contendrá las muestras separadas por clase
data_per_class = []

# Añadimos una lista vacía por clase
for _ in range(7):         
    data_per_class.append([])
# Añadimos a la lista de cada clase las muestras de esta
for sample in data:
    data_per_class[int(sample[54])].append(sample)



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
        line = line.replace('\n', '')
        line = line.split('|')
        if line[54] in categorical_encoder_catastral:
            line[54] = categorical_encoder_catastral[line[54]]
            if line[54] is 50:
                line[53] = -1
        data_predict.append(line)

# Finalmente convertimos las muestras preprocesadas a una matriz (no númerica, nos interesa el id esta vez)
data_predict = np.array(data_predict)

# Variable que contendra las predicciones globales de cada muestra
predictions = {}

# Número de iteraciones total por módelo
iterations = 50

# Variable anterior, inicializada de nuevo
predictions = {}

# Si True, muestra información de cada modelo local tras entrenarlo
debug_mode = True

# Variable en el rango (0.0 - 1.0) que indica el procentaje de muestras de validación
test_avg = 0.14

# Variable en el rango (0.0 - 1.0) que indica el procentaje de mejores modelos a utilizar
best_model_avg = 0.4

accuracy_avg = 0
precision_avg = 0
recall_avg = 0
f1_avg = 0

# Lista que contendra diccionarios con las metricas de cada modelo, predicciones y conjunto de datos utilizados
concensus = []

# Los diccionarios anteriores seguiran el siguiente formato:
'''
model_info = {
    "accuracy":
    "precision":
    "recall":
    "f1":
    "predictions":
    "data": {
            'X_train': X_train, 
            'X_test': X_test, 
            'y_train': y_train, 
            'y_test': y_test,
        },
    "model": <- Solo si 'persistent_mode' es True
}
'''

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

    # Mostramos el porcentaje de entrenamiento
    print('Entrenamiento completo al {}%'.format(ite/iterations * 100))

    np.random.shuffle(data_proc)
    X_train, X_test, y_train, y_test = train_test_split(data_proc[:, :54], data_proc[:, 54], test_size = test_avg)
    
    # Modelo XGB
    model = xgb.XGBClassifier(max_depth=None, learning_rate=0.1, n_estimators=400, verbosity=None, objective=None, 
        booster=None, tree_method=None, gamma=None, min_child_weight=None, max_delta_step=None, 
        subsample=None, colsample_bytree=None, colsample_bylevel=None, colsample_bynode=None, reg_alpha=None, 
        reg_lambda=None, scale_pos_weight=None, base_score=None, random_state=None)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Métricas del modelo entrenado
    if debug_mode:
        print('XGBoost ({})'.format(ite))
        print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))
        print('Precision (macro): {}'.format(precision_score(y_test, y_pred, average = 'macro')))
        print('Recall (macro): {}'.format(recall_score(y_test, y_pred, average = 'macro')))
        print('F1 (macro): {}\n'.format(f1_score(y_test, y_pred, average = 'macro')))
    
    # Actualización de las métricas de ENTRENAMIENTO
    accuracy_avg += accuracy_score(y_test, y_pred)
    precision_avg += precision_score(y_test, y_pred, average = 'macro')
    recall_avg += recall_score(y_test, y_pred, average = 'macro')
    f1_avg += f1_score(y_test, y_pred, average = 'macro')

    # Diccionario con la información del modelo
    model_info = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average = 'macro'),
        "recall": recall_score(y_test, y_pred, average = 'macro'),
        "f1": f1_score(y_test, y_pred, average = 'macro'),
        "predictions": model.predict(data_predict[:, 1:].astype('float32')),
        "data": {
            'X_train': X_train, 
            'X_test': X_test, 
            'y_train': y_train, 
            'y_test': y_test,
        },
    }
    concensus.append(model_info)
        
    # Modelo RandomForest
    model = RandomForestClassifier(n_estimators=400, criterion='entropy', max_depth=60, min_samples_split=5, 
        min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='log2', max_leaf_nodes=None, 
        min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-1, 
        random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Métricas del modelo entrenado
    if debug_mode:
        print('RF ({})'.format(ite))
        print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))
        print('Precision (macro): {}'.format(precision_score(y_test, y_pred, average = 'macro')))
        print('Recall (macro): {}'.format(recall_score(y_test, y_pred, average = 'macro')))
        print('F1 (macro): {}\n'.format(f1_score(y_test, y_pred, average = 'macro')))
    
    # Actualización de las métricas de ENTRENAMIENTO
    accuracy_avg += accuracy_score(y_test, y_pred)
    precision_avg += precision_score(y_test, y_pred, average = 'macro')
    recall_avg += recall_score(y_test, y_pred, average = 'macro')
    f1_avg += f1_score(y_test, y_pred, average = 'macro')

    # Diccionario con la información del modelo
    model_info = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average = 'macro'),
        "recall": recall_score(y_test, y_pred, average = 'macro'),
        "f1": f1_score(y_test, y_pred, average = 'macro'),
        "predictions": model.predict(data_predict[:, 1:].astype('float32')),
        "data": {
            'X_train': X_train, 
            'X_test': X_test, 
            'y_train': y_train, 
            'y_test': y_test,
        },
    }
    concensus.append(model_info)
            
print('\nEntrenamiento completo\n')
print('MÉTRICAS DEL ENTRENAMIENTO (global)')
print('Accuracy: {}'.format(accuracy_avg / (iterations * 2)))
print('Precision (macro): {}'.format(precision_avg / (iterations * 2)))
print('Recall (macro): {}'.format(recall_avg / (iterations * 2)))
print('F1(macro): {}'.format(f1_avg / (iterations * 2)))

# 1. Ordenamos 'concensous' según una métrica
concensus = sorted(concensus, key = lambda i: i['f1'], reverse = True)

# 2. Obtenemos los 'x' mejores modelos
n = int(iterations * 2 * best_model_avg)

# 3. Calculamos la métrica general para los 'x' modelos y predecimos
accuracy_avg = 0
precision_avg = 0
recall_avg = 0
f1_avg = 0

for i in range(n):
    # Métricas
    accuracy_avg += concensus[i]['accuracy']
    precision_avg += concensus[i]['precision']
    recall_avg += concensus[i]['recall']
    f1_avg += concensus[i]['f1']

    # Predicciones
    predictions_aux = concensus[i]['predictions']
    for i in range(len(data_predict)):
        if (data_predict[i, 0] not in predictions):
            predictions[data_predict[i, 0]] = [int(predictions_aux[i])]
        else:
            predictions[data_predict[i, 0]].append(int(predictions_aux[i]))

print('\nMÉTRICAS DEL MODELO (concenso)')
print('Accuracy: {}'.format(accuracy_avg / n))
print('Precision (macro): {}'.format(precision_avg / n))
print('Recall (macro): {}'.format(recall_avg / n))
print('F1 (macro): {}'.format(f1_avg / n))


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

with open(r'Resultados\Res_BASE.txt', 'w') as write_file:
    write_file.write('ID|CLASE\n')
    for sample in data_predict:
        write_file.write('{}|{}\n'.format(sample[0], categorical_decoder_class[most_frequent(predictions[sample[0]])]))