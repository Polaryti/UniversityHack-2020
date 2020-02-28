import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd


class Procesador:
    # Matriz que contendra los datos obtenidos del fichero de entrada
    raw_data = []
    raw_numeric_data = []
    # Matriz que contendra los datos tratados obtenidos raw_data
    input_data = []
    # Entero que indica el número de diferentes clases a predecir
    num_class = -1
    # Elemento delimitador para separar cada dato del fichero de entrada
    delimiter = '|'
    # Decimal que representa el porcentaje de muestras para evaluar el modelo [0.1 - 0.9]
    test_percentage = 0.1

    variables_name = []                 # Lista que contiene los nombres de las variables
    dictio = {'RESIDENTIAL\n': 0,
    'INDUSTRIAL\n': 1,
    'PUBLIC\n': 2,
    'OFFICE\n': 3,
    'RETAIL\n': 4,
    'OTHER\n': 5,
    'AGRICULTURE\n': 6}

    dictio_i = {0: 'RESIDENTIAL\n',
    1: 'INDUSTRIAL\n',
    2: 'PUBLIC\n',
    3: 'OFFICE\n',
    4: 'RETAIL\n',
    5: 'OTHER\n',
    6: 'AGRICULTURE\n'}

    def __init__(self, path: str, test_percentage: float):
        with open(path, 'r') as file:
            self.variables_name = file.readline().split(self.delimiter)

            for line in file.readlines():
                aux = line.split(self.delimiter)
                self.raw_data.append(aux)
                n_55 = aux[len(aux) - 1]
                n_54 = aux[len(aux) - 2]
                n_53 = aux[len(aux) - 3]
                aux = aux[1:len(aux) - 3]

                if n_54.isalpha() or n_54 == '""':
                    aux.append(0) # A Cambiar
                else:
                    aux.append(n_54)

                if n_54 == '""':
                    break
                    aux.append(-1) # A Cambiar
                else:
                    aux.append(n_53)

                aux.append(self.dictio[n_55])
                self.raw_numeric_data.append(aux)


        #self.input_data = np.array(self.raw_numeric_data).astype(float)
        self.input_data = self.raw_numeric_data
        self.__normalize_data()
        self.input_data = np.array(self.raw_numeric_data).astype(float)
        self.num_class = len(self.dictio)
        self.test_percentage = test_percentage

    # Devuelve una tupla de dos tensores, los datos y la clases a predecir
    def get_data(self):
        t_data = tf.constant(                                       # Tensor de rango 2 (matriz) que contiene los datos
            value=self.input_data[:, :len(self.input_data[0]) - 2]
        )
        t_label = tf.constant(                                      # Tensor de rango 1 (vector) que contiene las predicciones
            value=self.input_data[:, len(self.input_data[0]) - 1]
        )
        # Devolvemos una dupla de tensores donde cada tensor ha sido transformado en uno concatenando sus elementos en el eje 0
        return (tf.concat(t_data, 0), tf.concat(t_label, 0))

    def __normalize_data(self):
        normalize_max = [-1000] * (len(self.input_data[0]) - 1)       # Array que contiene el máximo valor de cada variable
        normalize_min = [99999999] * (len(self.input_data[0]) - 1)  # Array que contiene el minimo valor de cada variable
        res = []                                                    # Array que contendra los datos normalizados

        # Calculamos el máximo y minimo de todas las variables
        for i in range(len(self.input_data)):
            for j in range(len(normalize_max)):
                aux = self.input_data[i]
                if float(aux[j]) > normalize_max[j]:
                    normalize_max[j] = float(aux[j])
                if float(aux[j]) < normalize_min[j]:
                    normalize_min[j] = float(aux[j]) 
        
        # Normalizamos cada variable en un supuesto rango de  [0, 1]
        for i in range(len(self.input_data)):
            aux = self.input_data[i]
            for j in range(len(aux) - 1):
                aux[j] = round((float(aux[j]) - normalize_min[j]) / (normalize_max[j] - normalize_min[j]), 12)
            res.append(aux)
                
        self.input_data = res


# 1: Obtención y procesamiento de datos
procesador = Procesador(r'Data\Modelar_UH2020.txt', 0.20)
(samples_data, samples_labels) = procesador.get_data()

# 2: Creación de la red neuronal
model = keras.Sequential([
    keras.layers.Dense(128, activation='sigmoid'),
    keras.layers.Dense(128, activation='sigmoid'),
    keras.layers.Dense(128, activation='sigmoid'),
    keras.layers.Dense(128, activation='sigmoid'),
    keras.layers.Dense(procesador.num_class, activation='softmax')
])

# 3: Función de optimización y metrica a optimizar
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 4: Entrenamiento del modelo
model.fit(
    x=samples_data,
    y=samples_labels,
    epochs=1,
    validation_split=procesador.test_percentage
)

# 5: Saca la estadistica de acierto de cada clase
predictions = model.predict(samples_data)
cont_total = [0] * len(procesador.dictio)
cont_acierto = [0] * len(procesador.dictio)
cont = 0
for pr in predictions:
    cont_total[int(samples_labels[cont])] += 1
    if np.argmax(pr) == int(samples_labels[cont]):
        cont_acierto[int(samples_labels[cont])] += 1
    cont += 1
cont = 0
for cl in cont_total:
    print(("Tasa de acierto de {} es {}. Hay {} muestras.").format(procesador.dictio_i[cont], 100 * round(cont_acierto[cont] / cl, 2), cl))
    cont += 1