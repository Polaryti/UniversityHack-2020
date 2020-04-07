import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.manifold import LocallyLinearEmbedding

data = np.genfromtxt(r'Data\Modelar_UH2020.csv', delimiter = ',') # Preprocesamiento previo (codificación categorias, '|' -> ','...)

###
variables_per_class = []
for i in range(7):         
    variables_per_class.append([])

for label in data:
    variables_per_class[int(label[55])].append(label)

# Lista de datos equilibrados
eq_data = []
for i in range(338):
    for j in range(7):
        eq_data.append(variables_per_class[j][i])
eq_data = np.array(eq_data)
X = eq_data[:, 1:55]

print(X.shape)
embedding = LocallyLinearEmbedding(n_components=10)
X_transformed = embedding.fit_transform(X)
print(X_transformed.shape)
print(X_transformed[1])