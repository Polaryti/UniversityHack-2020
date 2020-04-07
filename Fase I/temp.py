import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
import tensorflow as tf
from tensorflow import keras

data = np.genfromtxt(r'Data\Modelar_UH2020.csv', delimiter = ',')
np.random.shuffle(data)
X_train_menor, X_test_menor, y_train_menor, y_test_menor = train_test_split(data[:, 1:55], data[:, 55], test_size = 0.1)

model = keras.Sequential([
    keras.layers.Dense(152, activation='relu'),
    keras.layers.Dense(7, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(
    x=X_train_menor,
    y=y_train_menor,
    epochs=1,
    validation_split=0.1,
    shuffle= True
)

res = model.predict(X_test_menor)
aux = []
for l in res:
    aux.append(np.argmax(l))
res = np.array(aux).astype('float32')
print(classification_report(y_test_menor, res))