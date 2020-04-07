import tensorflow as tf
from tensorflow import keras
import numpy as np
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
                for j in range(6):
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
        t_data_pos = tf.constant(                                   # Tensor de rango 2 (matriz) que contiene los datos de posición
             value=self.eq_input[:, 0:2]
        )

        # t_data_geo = tf.constant(                                   # Tensor de rango 2 (matriz) que contiene los datos de geometria
        #     value=self.eq_input[:, 47:55]
        # )

        # t_data_img = tf.constant(                                   # Tensor de rango 2 (matriz) que contiene los datos de imagen
        #     value=self.eq_input[:, 2:46]
        # )

        t_data_red = tf.constant(                                   # Tensor de rango 2 (matriz) que contiene los datos de imagen
            value=self.eq_input[:, 2:13]
        )
        t_data_gre = tf.constant(                                   # Tensor de rango 2 (matriz) que contiene los datos de imagen
            value=self.eq_input[:, 13:24]
        )
        t_data_blu = tf.constant(                                   # Tensor de rango 2 (matriz) que contiene los datos de imagen
            value=self.eq_input[:, 24:35]
        )
        t_data_inf = tf.constant(                                   # Tensor de rango 2 (matriz) que contiene los datos de imagen
            value=self.eq_input[:, 35:46]
        )

        t_label = tf.constant(                                      # Tensor de rango 1 (vector) que contiene las clases a predecir
            value=self.eq_labels
        )
        
        # Devolvemos una dupla de tensores donde cada tensor ha sido transformado en uno concatenando sus elementos en el eje 0
        return (t_data_pos, tf.concat(t_data_red, 0), tf.concat(t_data_gre, 0), tf.concat(t_data_blu, 0), tf.concat(t_data_inf, 0), tf.concat(t_label, 0))

# 1: Obtención y procesamiento de datos
procesador = Parser(r'Data\Modelar_UH2020.txt', 0.14)
(sample_data, red_data, gre_data, blu_data, inf_data, samples_labels) = procesador.get_data()

# 2: Creación de la red neuronal
# model = keras.Sequential([
#     keras.layers.Dense(128, activation='sigmoid'),
#     keras.layers.Dense(128, activation='sigmoid'),
#     keras.layers.Dense(128, activation='sigmoid'),
#     keras.layers.Dense(128, activation='sigmoid'),
#     keras.layers.Dense(procesador.num_class, activation='softmax')
# ])

model = tf.keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(procesador.num_class, activation='softmax')
])
# No converge > 30%
model_img_red = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(procesador.num_class, activation='softmax')
])

# No converge > 30%
model_img_gre = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(procesador.num_class, activation='softmax')
])

# Converge rapido 10%
model_img_blu = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(procesador.num_class, activation='softmax')
])

# Converge rapido 10%
model_img_inf = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(procesador.num_class, activation='softmax')
])

# 3: Función de optimización, función de perdidas y metrica a optimizar
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_img_red.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_img_gre.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_img_blu.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_img_inf.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4: Entrenamiento del modelo
model.fit(
    x=sample_data,
    y=samples_labels,
    epochs=100,
    validation_split=procesador.test_percentage,
    shuffle= True
)

# model_img_red.fit(
#     x=red_data,
#     y=samples_labels,
#     epochs=500,
#     validation_split=procesador.test_percentage,
#     shuffle= True
# )

# model_img_gre.fit(
#     x=gre_data,
#     y=samples_labels,
#     epochs=200,
#     validation_split=procesador.test_percentage,
#     shuffle= True
# )

# model_img_blu.fit(
#     x=blu_data,
#     y=samples_labels,
#     epochs=200,
#     validation_split=procesador.test_percentage,
#     shuffle= True
# )

# model_img_inf.fit(
#     x=inf_data,
#     y=samples_labels,
#     epochs=500,
#     validation_split=procesador.test_percentage,
#     shuffle= True
# )

# 5: Validar predicciones
# predictions_red = model_img_red.predict(red_data)
# predictions_gre = model_img_gre.predict(gre_data)
# predictions_blu = model_img_blu.predict(blu_data)
# predictions_inf = model_img_inf.predict(inf_data)

# cont = 0
# for i in range(20):
#     print(("C:{} ->:{}\n").format(samples_labels[cont], np.argmax(predictions_red[i] + predictions_gre[i] + predictions_blu[i] + predictions_inf[i])))
#     cont += 1