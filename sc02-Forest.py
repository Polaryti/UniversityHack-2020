import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import LocallyLinearEmbedding, SpectralEmbedding
from sklearn.cluster import MeanShift

data = np.genfromtxt(r'Data\Modelar_UH2020.csv', delimiter = ',') # Preprocesamiento previo (codificación categorias, '|' -> ','...)
predict = np.genfromtxt(r'Data\Estimar_UH2020.csv', delimiter='|')

dictio_i = {0: 'RESIDENTIAL\n',
    1: 'INDUSTRIAL\n',
    2: 'PUBLIC\n',
    3: 'OFFICE\n',
    4: 'OTHER\n',
    5: 'RETAIL\n',
    6: 'AGRICULTURE\n'}

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
X_train, X_test, y_train, y_test = train_test_split(eq_data[:, 1:55], eq_data[:, 55], test_size = 0.18)

###

###
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
###

###
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

final_predictions = clf.predict(predict[:, 1:])
with open('res-forest.txt', 'w') as file:
    with open(r'Data\Estimar_UH2020.csv', 'r') as read:
        for i in range(len(final_predictions)):
            file.write('{}|{}'.format(read.readline().split('|')[0], dictio_i[final_predictions[i]]))
###