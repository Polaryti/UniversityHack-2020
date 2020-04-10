from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import sampling
import random
import numpy as np
from sampling import tomeklinks, aiiknn

# Sección datos
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

print(data.shape)
pos = len(data[0]) - 1
X, Y = aiiknn(data[:, :pos], data[:, pos])
print(X.shape)



# Sección Grid Search 
estimator = XGBClassifier(
    random_state = 420,
    objective = 'binary:logistic',
    tree_method = 'gpu_hist',
    eval_metric = "logloss",
    nthread = -1,                     # 4 -> -1: Para utilizar todos los disponibles
    seed = 42,
)

#Generate lists of floats for each parameter
def gen(start, stop, increment):
  gen_list = []
  while start < stop:
    gen_list.append(start)
    start += increment
  return gen_list

params = [(0.0, 0.4, 0.2), (0.2, 0.6, 0.2), (2, 10, 2), (250, 1000, 250), (0.25, 1.0, 0.25), (0.25, 1.0, 0.25), (0.0, 0.4, 0.2), (0.5, 2.0, 0.5)]
res = []
for i in range(len(params)):
  res.append(gen(params[i][0], params[i][1], params[i][2]))

for i in res:
  print(i)

parameters = {
    'colsample_bytree' : res[0],
    'subsample' : res[1],
    'max_depth': res[2],
    'n_estimators': res[3],
    'learning_rate': res[4],
    'gamma' : res[5],
    'reg_alpha' : res[6],
    'reg_lambda' : res[7]
}

grid_search = GridSearchCV(
    estimator = estimator,
    param_grid = parameters,
    scoring = 'roc_auc',
    n_jobs = -1,              # 15 -> -1: Para utilizar todos los disponibles
    cv = 10,
    verbose = True
)

grid_search.fit(X, Y)

print(grid_search.best_estimator_)