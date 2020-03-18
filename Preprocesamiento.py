categorical_encoder_class = {'RESIDENTIAL\n': 0,
    'INDUSTRIAL\n': 1,
    'PUBLIC\n': 2,
    'OFFICE\n': 3,
    'OTHER\n': 4,
    'RETAIL\n': 5,
    'AGRICULTURE\n': 6
}

categorical_encoder_catastral = {'A': -10,
    'B': -20,
    'C': -30,
    '""': 50
}

import numpy as np

data = []
with open(r'Data\Modelar_UH2020.txt') as read_file:
    # La primera linea del documento es el nombre de las variables, no nos interesa
    read_file.readline()
    # Leemos línea por línea adaptando las muestras al formato deseado
    for line in read_file.readlines():
        line = line.split('|')
        if line[54] in categorical_encoder_catastral:
            line[54] = categorical_encoder_catastral[line[54]]
            if line[54] is 50:
                line[53] = -1
        line[55] = categorical_encoder_class[line[55]]
        # No nos interesa el identificador de la muestra
        data.append(line[1:])

# Finalmente convertimos las muestras preprocesadas a una matriz de números
data = np.array(data).astype('float32')