'''
En este modelo se ha realizado:
- CONSTRUCTIONYEAR -> Cambiar por antiguedad (NO CAMBIA NADA)
- MAXBUILDINGFLOOR -> Las entradas null de -1 a 0 (NO CAMBIA NADA)
- CADASTRALQUALITYID -> Transformar a one-hot

Y se implementa:
- Support Vector Machine, con datos balenceados
'''
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score
import random
from sklearn import svm
from imblearn.under_sampling import TomekLinks

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
# data_per_class = []

# # Añadimos una lista vacía por clase
# for _ in range(7):         
#     data_per_class.append([])
# # Añadimos a la lista de cada clase las muestras de esta
# for sample in data:
#     data_per_class[int(sample[len(sample) - 1])].append(sample)

# # Variable que contendrá los datos procesados
# data_proc = []

# Variable en el rango (0.0 - 1.0) que indica el procentaje de muestras de validación
test_avg = 0.2

# # Muestras de la clase RESIDENTIAL
# random.shuffle(data_per_class[0])
# data_proc += data_per_class[0][:5500]

# # Muestras de las otras clases
# for i in range(6):
#    data_proc += data_per_class[i + 1]
        
# # Volvemos a convertir los datos una vez procesados a una matriz
# data_proc = np.array(data_proc)

# np.random.shuffle(data_proc)

# X -> Datos ya tratados sin predicción
# Y -> Predicción
last_position = len(data[0]) - 1
X, Y = (data[:, :last_position], data[:, last_position])

tl = TomekLinks()
X, Y = tl.fit_resample(X, Y)

sss = StratifiedShuffleSplit(
    n_splits = 1,       # Solo una partición
    test_size = 0.2,    # Repartición 80/20 
)

# La función es un iterador (debemos iterar, aunque sea solo una vez)
for train_index, test_index in sss.split(X, Y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]


clf = svm.SVC(
    C = 0.001,
    #kernel = 'poly',
    degree = 6,
    gamma = 0.01,                     # ¿Por que Jon dijo 0 si no es un valor valido?
    decision_function_shape = 'ovo',
    )


# print(cross_val_score(
#    clf,  X, Y, cv = 10, scoring = 'f1_macro'))

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))