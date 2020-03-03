import tensorflow as tf
from tensorflow import keras
import numpy as np
import tempfile

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

    batch_size = 128
    epochs = 20

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
        porcentaje = int(len(self.input_data) * 0.2)
        t_data = tf.constant(                                       # Tensor de rango 2 (matriz) que contiene los datos
            value=self.input_data[0:porcentaje, :len(self.input_data[0]) - 2]
        )
        t_labels = keras.utils.to_categorical(self.input_data[0:porcentaje, len(self.input_data[0]) - 1], procesador.num_class)
        test_data = tf.constant(                                       # Tensor de rango 2 (matriz) que contiene los datos
            value=self.input_data[porcentaje:len(self.input_data), :len(self.input_data[0]) - 2]
        )
        test_labels = keras.utils.to_categorical(self.input_data[porcentaje:len(self.input_data), len(self.input_data[0]) - 1], procesador.num_class)


        # Devolvemos una dupla de tensores donde cada tensor ha sido transformado en uno concatenando sus elementos en el eje 0
        return (t_data, t_labels, test_data, test_labels)

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
logdir = tempfile.mkdtemp()
procesador = Procesador(r'Data\Modelar_UH2020.txt', 0.20)
(samples_data, samples_labels, test_data, test_labels) = procesador.get_data()

# 2: Creación de la red neuronal
end_step = np.ceil(1.0 * samples_data.shape[0] / procesador.batch_size).astype(np.int32) * procesador.epochs
pruning_params = {
      'pruning_schedule': keras.PolynomialDecay(initial_sparsity=0.50,
                                                   final_sparsity=0.90,
                                                   begin_step=2000,
                                                   end_step=end_step,
                                                   frequency=100)
}

pruned_model = keras.Sequential([
    keras.prune_low_magnitude(
        keras.layers.Conv2D(32, 5, padding='same', activation='relu'),
        **pruning_params),
    keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same'),
    keras.layers.BatchNormalization(),
    keras.prune_low_magnitude(
        keras.layers.Conv2D(64, 5, padding='same', activation='relu'), **pruning_params),
    keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same'),
    keras.layers.Flatten(),
    keras.prune_low_magnitude(keras.layers.Dense(1024, activation='relu'),
                                 **pruning_params),
    keras.layers.Dropout(0.4),
    keras.prune_low_magnitude(keras.layers.Dense(procesador.num_class, activation='softmax'),
                                 **pruning_params)
])

# 3: Función de optimización y metrica a optimizar
pruned_model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer='adam',
    metrics=['accuracy'])

callbacks = [
    keras.UpdatePruningStep(),
    keras.PruningSummaries(log_dir=logdir, profile_batch=0)
]

# 4: Entrenamiento del modelo
pruned_model.fit(samples_data, samples_labels,
          batch_size = procesador.batch_size,
          epochs = procesador.epochs,
          verbose = 1,
          callbacks = callbacks,
          validation_data = (test_data, test_labels))

# 5: Saca la estadistica de acierto de cada clase
score = pruned_model.evaluate(test_data, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])