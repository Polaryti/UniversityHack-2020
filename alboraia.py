import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

data = np.genfromtxt(r'Data\Modelar_UH2020.csv', delimiter = ',')

np.random.shuffle(data)
variables_per_class = []
for i in range(7):         
    variables_per_class.append([])
for label in data:
    variables_per_class[int(label[55])].append(label)

    ## Data normalizada al por menor
eq_data_menor = []
    # for i in range(338):
    #     for j in range(7):
    #         if j is 6:
    #             eq_data_menor.append(variables_per_class[j][i % 338])
    #         else:
    #             eq_data_menor.append(variables_per_class[j][i])
    # eq_data_menor = np.array(eq_data_menor)
for i in range(4490):
    eq_data_menor.append(variables_per_class[0][i])
eq_data_menor += variables_per_class[1]
eq_data_menor += variables_per_class[2]
eq_data_menor += variables_per_class[3]
eq_data_menor += variables_per_class[4]
eq_data_menor += variables_per_class[5]
eq_data_menor += variables_per_class[6]
eq_data_menor = np.array(eq_data_menor)


model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(7, activation='softmax')
])

# 3: Función de optimización y metrica a optimizar
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['categorical_accuracy'])

# 4: Entrenamiento del modelo
model.fit(
    x=eq_data_menor[:, 1:55],
    y=eq_data_menor[:, 55],
    epochs=5,
    validation_split=0.2
)