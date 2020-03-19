import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

# Muestras de la clase RESIDENTIAL
data_proc += data_per_class[0][0:16000]
# Muestras de las otras clases
for i in range(6):
    data_proc += data_per_class[i + 1]
    
# Volvemos a convertir los datos una vez procesados a una matriz
data_proc = np.array(data_proc)

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
test_avg = 0.18

for ite in range(iterations):
    # Mostramos el porcentaje de entrenamiento
    print('Entrenamiento completo al {}%'.format(ite/iterations * 100))

    np.random.shuffle(data_proc)
    X_train, X_test, y_train, y_test = train_test_split(data_proc[:, :54], data_proc[:, 54], test_size = test_avg)
    
    # Modelo XGB
    model = xgb.XGBClassifier(max_depth=None, learning_rate=0.1, n_estimators=400, verbosity=None, objective=None, 
        booster=None, tree_method=None, n_jobs=-1, gamma=None, min_child_weight=None, max_delta_step=None, 
        subsample=None, colsample_bytree=None, colsample_bylevel=None, colsample_bynode=None, reg_alpha=None, 
        reg_lambda=None, scale_pos_weight=None, base_score=None, random_state=None)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if debug_mode:
        #print('Matriz de confusión:\n{}\n'.format(confusion_matrix(y_test, y_pred)))
        print('Informe de clasificación:\n{}\n'.format(classification_report(y_test, y_pred)))
        print(accuracy_score(y_test, y_pred))
    
    predictions_aux = model.predict(data_predict[:, 1:].astype('float32'))
    for i in range(len(data_predict)):
        if (data_predict[i, 0] not in predictions):
            predictions[data_predict[i, 0]] = [int(predictions_aux[i])]
        else:
            predictions[data_predict[i, 0]].append(int(predictions_aux[i]))
        
    # Modelo RandomForest
    model = RandomForestClassifier(n_estimators=400, criterion='entropy', max_depth=60, min_samples_split=5, 
        min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='log2', max_leaf_nodes=None, 
        min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-1, 
        random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if debug_mode:
        # print('Matriz de confusión:\n{}\n'.format(confusion_matrix(y_test, y_pred)))
        print('Informe de clasificación:\n{}\n'.format(classification_report(y_test, y_pred)))
        print(accuracy_score(y_test, y_pred))
    
    predictions_aux = model.predict(data_predict[:, 1:].astype('float32'))
    for i in range(len(data_predict)):
        if (data_predict[i, 0] not in predictions):
            predictions[data_predict[i, 0]] = [int(predictions_aux[i])]
        else:
            predictions[data_predict[i, 0]].append(int(predictions_aux[i]))
print('Entrenamiento completo')

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

with open(r'Minsait_Universitat Politècnica de València_Astralaria.txt', 'w') as write_file:
    write_file.write('ID|CLASE\n')
    for sample in data_predict:
        write_file.write('{}|{}\n'.format(sample[0], categorical_decoder_class[most_frequent(predictions[sample[0]])]))