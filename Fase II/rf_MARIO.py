import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
import xgboost as xgb
from sklearn.metrics import classification_report, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import random
import random
import pandas as pd

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

data_proc = []

# Muestras de la clase RESIDENTIAL
random.shuffle(data_per_class[0])
data_proc += data_per_class[0][:5250]

# Muestras de las otras clases
for i in range(6):
    data_proc += data_per_class[i + 1]
        
 # Volvemos a convertir los datos una vez procesados a una matriz
data_proc = np.array(data_proc)

# Realizamos under_sampling
pos = len(data_proc[0]) - 1
X, Y = (data_proc[:, :pos], data_proc[:, pos])


# Modelo RF
model = RandomForestClassifier(
    criterion = 'entropy',
    n_jobs = -1,
    max_features = None,
    n_estimators = 400,
    max_depth = 50,
    min_samples_split = 3,
    min_samples_leaf = 1,
)

parameters = {
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3],
}

grid_search = GridSearchCV(
    estimator = model,
    param_grid = parameters,
    scoring = 'f1_macro',
    n_jobs = -1,
    cv = 4,
    verbose = True,
)

grid_search.fit(X, Y)

print(grid_search.best_params_)

# Evaluación de las variables con unos parametros
columns = 'X|Y|Q_R_4_0_0|Q_R_4_0_1|Q_R_4_0_2|Q_R_4_0_3|Q_R_4_0_4|Q_R_4_0_5|Q_R_4_0_6|Q_R_4_0_7|Q_R_4_0_8|Q_R_4_0_9|Q_R_4_1_0|Q_G_3_0_0|Q_G_3_0_1|Q_G_3_0_2|Q_G_3_0_3|Q_G_3_0_4|Q_G_3_0_5|Q_G_3_0_6|Q_G_3_0_7|Q_G_3_0_8|Q_G_3_0_9|Q_G_3_1_0|Q_B_2_0_0|Q_B_2_0_1|Q_B_2_0_2|Q_B_2_0_3|Q_B_2_0_4|Q_B_2_0_5|Q_B_2_0_6|Q_B_2_0_7|Q_B_2_0_8|Q_B_2_0_9|Q_B_2_1_0|Q_NIR_8_0_0|Q_NIR_8_0_1|Q_NIR_8_0_2|Q_NIR_8_0_3|Q_NIR_8_0_4|Q_NIR_8_0_5|Q_NIR_8_0_6|Q_NIR_8_0_7|Q_NIR_8_0_8|Q_NIR_8_0_9|Q_NIR_8_1_0|AREA|GEOM_R1|GEOM_R2|GEOM_R3|GEOM_R4|CONTRUCTIONYEAR|MAXBUILDINGFLOOR|CADASTRALQUALITYID_00|CADASTRALQUALITYID_01|CADASTRALQUALITYID_02|CADASTRALQUALITYID_03|CADASTRALQUALITYID_04|CADASTRALQUALITYID_05|CADASTRALQUALITYID_06|CADASTRALQUALITYID_07|CADASTRALQUALITYID_08|CADASTRALQUALITYID_09|CADASTRALQUALITYID_10|CADASTRALQUALITYID_11|CADASTRALQUALITYID_12'.split('|')
var_inf = pd.DataFrame({
        'Variable': columns,
        'Importance': grid_search.best_estimator_.feature_importances_
    }).sort_values('Importance', ascending = False)

print(var_inf)