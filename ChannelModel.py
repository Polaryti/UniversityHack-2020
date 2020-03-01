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

    # Listas equilibradas
    eq_input = []
    eq_labels = []

    # Lista por deciles
    data_dec_0 = []
    data_dec_1 = []
    data_dec_2 = []
    data_dec_3 = []
    data_dec_4 = []
    data_dec_5 = []
    data_dec_6 = []
    data_dec_7 = []
    data_dec_8 = []
    data_dec_9 = []
    data_dec_10 = []

    num_class = 7
    delimiter = '|'
    test_percentage = 1.0

    def __init__(self, path: str, test_percentage: float):
        # Inicializar las listas por clases
        for i in range(7):
            self.id_per_class.append([])           
            self.variables_per_class.append([])    
            self.labels_per_class.append([]) 
        
        with open(path, 'r') as file:
            # Obtenemos el nombre de las variables
            self.variables_name = file.readline().split(self.delimiter)

            for aux in file.readlines():
                # Separamos cada variable
                line = aux.split(self.delimiter)
                # Preprocesamiento simple
                line = self.__simple_prep(line)

                # Separamos cada muestra por clase
                self.id_per_class[self.class_to_number[line[len(line) - 1]]].append(line[0])
                self.variables_per_class[self.class_to_number[line[len(line) - 1]]].append(line[1:len(line) - 1])
                self.labels_per_class[self.class_to_number[line[len(line) - 1]]].append(self.class_to_number[line[len(line) - 1]])
            
            # Optimizamos las matrices
            for i in range(7):
                self.variables_per_class[i] = np.array(self.variables_per_class[i]).astype(float)

            # Lista de datos equilibrados
            for i in range(338):
                for j in range(7):
                    self.eq_input.append(self.variables_per_class[j][i])
                    self.eq_labels.append(self.labels_per_class[j][i])
            self.eq_input = np.array(self.eq_input).astype(float)
            self.eq_labels = np.array(self.eq_labels).astype(float)

            # Mezclamos los datos
            random.shuffle(self.eq_input, random.seed(103))
            random.shuffle(self.eq_labels, random.seed(103))

            # Normalizamos los datos
            self.__normalize_data

            # Lista de deciles
            for var in self.eq_input:
                aux = []
                aux.append(var[2])
                aux.append(var[13])
                aux.append(var[24])
                aux.append(var[35])
                self.data_dec_0.append(aux)
                aux = []
                aux.append(var[3])
                aux.append(var[14])
                aux.append(var[25])
                aux.append(var[36])
                self.data_dec_1.append(aux)
                aux = []
                aux.append(var[4])
                aux.append(var[15])
                aux.append(var[26])
                aux.append(var[37])
                self.data_dec_2.append(aux)
                aux = []
                aux.append(var[5])
                aux.append(var[16])
                aux.append(var[27])
                aux.append(var[38])
                self.data_dec_3.append(aux)
                aux = []
                aux.append(var[6])
                aux.append(var[17])
                aux.append(var[28])
                aux.append(var[39])
                self.data_dec_4.append(aux)
                aux = []
                aux.append(var[7])
                aux.append(var[18])
                aux.append(var[29])
                aux.append(var[40])
                self.data_dec_5.append(aux)
                aux = []
                aux.append(var[8])
                aux.append(var[19])
                aux.append(var[30])
                aux.append(var[41])
                self.data_dec_6.append(aux)
                aux = []
                aux.append(var[9])
                aux.append(var[20])
                aux.append(var[31])
                aux.append(var[42])
                self.data_dec_7.append(aux)
                aux = []
                aux.append(var[10])
                aux.append(var[21])
                aux.append(var[32])
                aux.append(var[43])
                self.data_dec_8.append(aux)
                aux = []
                aux.append(var[11])
                aux.append(var[22])
                aux.append(var[33])
                aux.append(var[44])
                self.data_dec_9.append(aux)
                aux = []
                aux.append(var[12])
                aux.append(var[23])
                aux.append(var[34])
                aux.append(var[45])
                self.data_dec_10.append(aux)


            self.data_dec_0 = np.array(self.data_dec_0).astype(float)
            self.data_dec_1 = np.array(self.data_dec_1).astype(float)
            self.data_dec_2 = np.array(self.data_dec_2).astype(float)
            self.data_dec_3 = np.array(self.data_dec_3).astype(float)
            self.data_dec_4 = np.array(self.data_dec_4).astype(float)
            self.data_dec_5 = np.array(self.data_dec_5).astype(float)
            self.data_dec_6 = np.array(self.data_dec_6).astype(float)
            self.data_dec_7 = np.array(self.data_dec_7).astype(float)
            self.data_dec_8 = np.array(self.data_dec_8).astype(float)
            self.data_dec_9 = np.array(self.data_dec_9).astype(float)
            self.data_dec_10 = np.array(self.data_dec_10).astype(float)
            
            self.test_percentage = test_percentage

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
        t_data_0 = tf.constant(                                   # Tensor de rango 2 (matriz) que contiene los datos de imagen
            value=self.data_dec_0
        )
        t_data_1 = tf.constant(                                   # Tensor de rango 2 (matriz) que contiene los datos de imagen
            value=self.data_dec_1
        )
        t_data_2 = tf.constant(                                   # Tensor de rango 2 (matriz) que contiene los datos de imagen
            value=self.data_dec_2
        )
        t_data_3 = tf.constant(                                   # Tensor de rango 2 (matriz) que contiene los datos de imagen
            value=self.data_dec_3
        )
        t_data_4 = tf.constant(                                   # Tensor de rango 2 (matriz) que contiene los datos de imagen
            value=self.data_dec_4
        )
        t_data_5 = tf.constant(                                   # Tensor de rango 2 (matriz) que contiene los datos de imagen
            value=self.data_dec_5
        )
        t_data_6 = tf.constant(                                   # Tensor de rango 2 (matriz) que contiene los datos de imagen
            value=self.data_dec_6
        )
        t_data_7 = tf.constant(                                   # Tensor de rango 2 (matriz) que contiene los datos de imagen
            value=self.data_dec_7
        )
        t_data_8 = tf.constant(                                   # Tensor de rango 2 (matriz) que contiene los datos de imagen
            value=self.data_dec_8
        )
        t_data_9 = tf.constant(                                   # Tensor de rango 2 (matriz) que contiene los datos de imagen
            value=self.data_dec_9
        )
        t_data_10 = tf.constant(                                   # Tensor de rango 2 (matriz) que contiene los datos de imagen
            value=self.data_dec_10
        )

        t_label = tf.constant(                                      # Tensor de rango 1 (vector) que contiene las clases a predecir
            value=self.eq_labels
        )
        
        # Devolvemos una dupla de tensores donde cada tensor ha sido transformado en uno concatenando sus elementos en el eje 0
        return (tf.concat(t_data_0, 0), tf.concat(t_data_1, 0), tf.concat(t_data_2, 0), tf.concat(t_data_3, 0), tf.concat(t_data_4, 0), tf.concat(t_data_5, 0), tf.concat(t_data_6, 0), tf.concat(t_data_7, 0), tf.concat(t_data_8, 0), tf.concat(t_data_9, 0), tf.concat(t_data_10, 0), tf.concat(t_label, 0))

# 1: Obtención y procesamiento de datos
procesador = Parser(r'Data\Modelar_UH2020.txt', 0.14)
(data0, data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, labels) = procesador.get_data()

# 2: Creación de la red neuronal
model_0 = keras.Sequential([
    keras.layers.Dense(16, activation='softsign'),
    keras.layers.Dense(procesador.num_class, activation='softmax')
])

model_1 = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(procesador.num_class, activation='softmax')
])

model_2 = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(procesador.num_class, activation='softmax')
])

model_3 = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(procesador.num_class, activation='softmax')
])

model_4 = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(procesador.num_class, activation='softmax')
])

model_5 = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(procesador.num_class, activation='softmax')
])

model_6 = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(procesador.num_class, activation='softmax')
])

model_7 = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(procesador.num_class, activation='softmax')
])

model_8 = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(procesador.num_class, activation='softmax')
])

model_9 = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(procesador.num_class, activation='softmax')
])

model_10 = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(procesador.num_class, activation='softmax')
])

# 3: Función de optimización, función de perdidas y metrica a optimizar
model_0.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_1.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_2.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_3.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_4.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_5.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_6.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_7.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_8.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_9.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_10.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



# 4: Entrenamiento del modelo
model_0.fit(
    x=data0,
    y=labels,
    epochs=120,
    validation_split=procesador.test_percentage,
    shuffle= True
)

model_1.fit(
    x=data1,
    y=labels,
    epochs=120,
    validation_split=procesador.test_percentage,
    shuffle= True
)

model_2.fit(
    x=data2,
    y=labels,
    epochs=120,
    validation_split=procesador.test_percentage,
    shuffle= True
)

model_3.fit(
    x=data3,
    y=labels,
    epochs=120,
    validation_split=procesador.test_percentage,
    shuffle= True
)

model_4.fit(
    x=data4,
    y=labels,
    epochs=120,
    validation_split=procesador.test_percentage,
    shuffle= True
)

model_5.fit(
    x=data5,
    y=labels,
    epochs=120,
    validation_split=procesador.test_percentage,
    shuffle= True
)

model_6.fit(
    x=data6,
    y=labels,
    epochs=120,
    validation_split=procesador.test_percentage,
    shuffle= True
)

model_7.fit(
    x=data7,
    y=labels,
    epochs=120,
    validation_split=procesador.test_percentage,
    shuffle= True
)

model_8.fit(
    x=data8,
    y=labels,
    epochs=120,
    validation_split=procesador.test_percentage,
    shuffle= True
)

model_9.fit(
    x=data9,
    y=labels,
    epochs=120,
    validation_split=procesador.test_percentage,
    shuffle= True
)

model_10.fit(
    x=data10,
    y=labels,
    epochs=120,
    validation_split=procesador.test_percentage,
    shuffle= True
)

# 5: Validar predicciones
predictions_0 = model_0.predict(data0)
predictions_1 = model_1.predict(data1)
predictions_2 = model_2.predict(data2)
predictions_3 = model_3.predict(data3)
predictions_4 = model_4.predict(data4)
predictions_5 = model_5.predict(data5)
predictions_6 = model_6.predict(data6)
predictions_7 = model_7.predict(data7)
predictions_8 = model_8.predict(data8)
predictions_9 = model_9.predict(data9)
predictions_10 = model_10.predict(data10)

cont = 0
for i in range(35):
    print(("C:{} ->:{}\n").format(labels[cont], np.argmax(predictions_0[i] + predictions_1[i] + predictions_2[i] + predictions_3[i] + predictions_4[i] + predictions_5[i] + predictions_6[i] + predictions_7[i] + predictions_8[i] + predictions_9[i] + predictions_10[i])))
    cont += 1