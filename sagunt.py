import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import neighbors

# read in data
data = np.genfromtxt(r'Data\Modelar_UH2020.csv', delimiter = ',')
predict = np.genfromtxt(r'Data\Estimar_UH2020.csv', delimiter='|')

dictio_i = {0: 'RESIDENTIAL\n',
    1: 'INDUSTRIAL\n',
    2: 'PUBLIC\n',
    3: 'OFFICE\n',
    4: 'RETAIL\n',
    5: 'OTHER\n',
    6: 'AGRICULTURE\n'}

variables_per_class = []
for i in range(7):         
    variables_per_class.append([])

for label in data:
    variables_per_class[int(label[55])].append(label)

eq_data = []
for i in range(90173):
    for j in range(7):
        eq_data.append(variables_per_class[j][i % (len(variables_per_class[j]) - 30)])

eq_test = []
for i in range(30):
    for j in range(7):
        eq_test.append(variables_per_class[j][(len(variables_per_class[j]) - 30) + i])


eq_data = np.array(eq_data)
eq_test = np.array(eq_test)

X_train = eq_data[:, 1:55]
y_train = eq_data[:, 55]
X_test = eq_test[:, 1:55]
y_test = eq_test[:, 55]

# specify parameters via map
nbrs = neighbors.KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)

# make prediction
y_pred = nbrs.predict(X_test)
print(classification_report(y_test, y_pred))

# file
final_predictions = nbrs.predict(predict[:, 1:])
with open('res-sagunt.txt', 'w') as file:
    with open(r'Data\Estimar_UH2020.csv', 'r') as read:
        for i in range(len(final_predictions)):
            file.write('{} - {}'.format(read.readline().split('|')[0], dictio_i[final_predictions[i]]))