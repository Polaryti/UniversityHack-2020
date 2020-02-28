import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import random


class Parser:
    class_to_number = {'RESIDENTIAL\n': 0,
                       'INDUSTRIAL\n': 1,
                       'PUBLIC\n': 2,
                       'OFFICE\n': 3,
                       'RETAIL\n': 4,
                       'OTHER\n': 5,
                       'AGRICULTURE\n': 6}

    class_to_name = {0: 'RESIDENTIAL\n',
                     1: 'INDUSTRIAL\n',
                     2: 'PUBLIC\n',
                     3: 'OFFICE\n',
                     4: 'RETAIL\n',
                     5: 'OTHER\n',
                     6: 'AGRICULTURE\n'}

    variables_name = []

    # Listas separadas
    id_per_class = []           # Acceder con class_to_number[]
    variables_per_class = []    # Acceder con class_to_number[]
    labels_per_class = []       # Acceder con class_to_number[]

    # Listas unitarias
    all_input = []              
    all_cord = []
    all_labels = []

    # Listas equilibradas
    eq_input = []
    eq_labels = []

    num_class = 7
    delimiter = '|'
    test_percentage = 1.0

    def __init__(self, path: str, test_percentage: float):
        for i in range(7):
            self.id_per_class.append([])           
            self.variables_per_class.append([])    
            self.labels_per_class.append([]) 
        
        with open(path, 'r') as file:
            # Obtenemos el nombre de las variables
            self.variables_name = file.readline().split(self.delimiter)

            # Lista de datos total + listas de datos por clases
            for aux in file.readlines():
                line = aux.split(self.delimiter)
                line = self.__simple_prep(line)

                self.all_input.append(line[1:len(line) - 1])
                self.all_cord.append([float(line[1]), float(line[2])])
                self.all_labels.append(float(self.class_to_number[line[len(line) - 1]]))

                self.id_per_class[self.class_to_number[line[len(line) - 1]]].append(line[0])
                self.variables_per_class[self.class_to_number[line[len(line) - 1]]].append(line[1:len(line) - 1])
                self.labels_per_class[self.class_to_number[line[len(line) - 1]]].append(self.class_to_number[line[len(line) - 1]])
            for i in range(7):
                self.variables_per_class[i] = np.array(self.variables_per_class[i]).astype(float)

            # Lista de datos equilibrados
            for i in range(338):
                for j in range(7):
                    self.eq_input.append(self.variables_per_class[j][i])
                    self.eq_labels.append(self.labels_per_class[j][i])
            random.shuffle(self.eq_input, random.seed(10))
            random.shuffle(self.eq_labels, random.seed(10))
            self.eq_input = np.array(self.eq_input).astype(float)
            self.eq_labels = np.array(self.eq_labels).astype(float)
            
            self.test_percentage = test_percentage
            self.__normalize_data

    def __simple_prep(self, aux: str):
        res = aux
        n_54 = aux[len(aux) - 2]
        n_53 = aux[len(aux) - 3]

        if n_54.isalpha() or n_54 == '""':
            res[len(res) - 2] = 0 # A Cambiar

        if n_54 == '""':
            res[len(res) - 3] = -1  # A Cambiar

        return res

    def __normalize_data(self):
        normalize_max = [-1000] * (len(self.eq_input[0]) - 1)       # Array que contiene el máximo valor de cada variable
        normalize_min = [99999999] * (len(self.eq_input[0]) - 1)  # Array que contiene el minimo valor de cada variable
        res = []                                                    # Array que contendra los datos normalizados

        # Calculamos el máximo y minimo de todas las variables
        for i in range(len(self.eq_input)):
            for j in range(len(normalize_max)):
                aux = self.eq_input[i]
                if float(aux[j]) > normalize_max[j]:
                    normalize_max[j] = float(aux[j])
                if float(aux[j]) < normalize_min[j]:
                    normalize_min[j] = float(aux[j]) 
        
        # Normalizamos cada variable en un rango [0, 1]
        for i in range(len(self.eq_input)):
            aux = self.eq_input[i]
            for j in range(len(aux) - 1):
                aux[j] = round((float(aux[j]) - normalize_min[j]) / (normalize_max[j] - normalize_min[j]), 12)
            res.append(aux)
                
        self.eq_input = res

    
    def get_data(self):
        t_data = tf.constant(                                       # Tensor de rango 2 (matriz) que contiene todos datos
            value=self.eq_input
        )
        
        t_data_pos = tf.constant(                                   # Tensor de rango 2 (matriz) que contiene los datos de posición
            value=self.eq_input[:, 0:2]
        )

        t_data_geo = tf.constant(                                   # Tensor de rango 2 (matriz) que contiene los datos de geometria
            value=self.eq_input[:, len(self.eq_input) - 8:len(self.eq_input)]
        )

        t_data_img = tf.constant(                                   # Tensor de rango 2 (matriz) que contiene los datos de imagen
            value=self.eq_input[:, 2:len(self.eq_input) - 8]
        )

        t_label = tf.constant(                                      # Tensor de rango 1 (vector) que contiene las clases a predecir
            value=self.eq_labels
        )
        
        # Devolvemos una dupla de tensores donde cada tensor ha sido transformado en uno concatenando sus elementos en el eje 0
        return (tf.concat(t_data, 0), tf.concat(t_data_pos, 0), tf.concat(t_data_geo, 0), tf.concat(t_data_img, 0), tf.concat(t_label, 0))


# 1: Obtención y procesamiento de datos
procesador = Parser(r'Data\Modelar_UH2020.txt', 0.1)
(samples_data, pos_data, geo_data, img_data, samples_labels) = procesador.get_data()

# 2: Creación de la red neuronal
model = keras.Sequential([
    keras.layers.Dense(128, activation='sigmoid'),
    keras.layers.Dense(128, activation='sigmoid'),
    keras.layers.Dense(128, activation='sigmoid'),
    keras.layers.Dense(128, activation='sigmoid'),
    keras.layers.Dense(procesador.num_class, activation='softmax')
])

model_pos = keras.Sequential([
    keras.layers.Dense(128, activation='sigmoid'),
    keras.layers.Dense(128, activation='sigmoid'),
    keras.layers.Dense(128, activation='sigmoid'),
    keras.layers.Dense(128, activation='sigmoid'),
    keras.layers.Dense(procesador.num_class, activation='softmax')
])

model_met = keras.Sequential([
    keras.layers.Dense(128, activation='sigmoid'),
    keras.layers.Dense(128, activation='sigmoid'),
    keras.layers.Dense(128, activation='sigmoid'),
    keras.layers.Dense(128, activation='sigmoid'),
    keras.layers.Dense(procesador.num_class, activation='softmax')
])

model_img = keras.Sequential([
    keras.layers.Dense(128, activation='sigmoid'),
    keras.layers.Dense(128, activation='sigmoid'),
    keras.layers.Dense(128, activation='sigmoid'),
    keras.layers.Dense(128, activation='sigmoid'),
    keras.layers.Dense(procesador.num_class, activation='softmax')
])

# 3: Función de optimización, función de perdidas y metrica a optimizar
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_pos.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_met.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_img.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4: Entrenamiento del modelo
model.fit(
    x=samples_data,
    y=samples_labels,
    epochs=35,
    validation_split=procesador.test_percentage,
    shuffle= True
)

model_pos.fit(
    x=pos_data,
    y=samples_labels,
    epochs=35,
    validation_split=procesador.test_percentage,
    shuffle= True
)

model_met.fit(
    x=geo_data,
    y=samples_labels,
    epochs=35,
    validation_split=procesador.test_percentage,
    shuffle= True
)

model_img.fit(
    x=img_data,
    y=samples_labels,
    epochs=35,
    validation_split=procesador.test_percentage,
    shuffle= True
)

# 5: Validar predicciones
predictions = model.predict(samples_data)
predictions_pos = model_pos.predict(pos_data)
predictions_geo = model_met.predict(geo_data)
predictions_img = model_img.predict(img_data)

cont = 0
for i in range(20):
    print(("C:{} ->:{}\n").format(samples_labels[cont], predictions[i]))
    print(("C:{} ->:{}\n").format(samples_labels[cont], predictions_pos[i]))
    print(("C:{} ->:{}\n").format(samples_labels[cont], predictions_geo[i]))
    print(("C:{} ->:{}\n").format(samples_labels[cont], predictions_img[i]))
    cont += 1