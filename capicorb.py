# Datos equilibrados al por mayor
# Requilibrado sin mezcla de muestras de entrenamiento en muestras de test 

import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

test_avg = 0.2

dictio_i = {0: 'RESIDENTIAL\n',
    1: 'INDUSTRIAL\n',
    2: 'PUBLIC\n',
    3: 'OFFICE\n',
    4: 'OTHER\n',
    5: 'RETAIL\n',
    6: 'AGRICULTURE\n'}

##
data = np.genfromtxt(r'Data\Modelar_UH2020.csv', delimiter = ',')
predict = np.genfromtxt(r'Data\Estimar_UH2020.csv', delimiter='|')

np.random.shuffle(data)
variables_per_class = []
for i in range(7):         
    variables_per_class.append([])
for label in data:
    variables_per_class[int(label[55])].append(label)

eq_data = []
for i in range(90173):
    for j in range(7):
        max_sample = len(variables_per_class[j]) - int(len(variables_per_class[j]) * test_avg)
        eq_data.append(variables_per_class[j][i % max_sample])

eq_test = []
for i in range(7):
    min_sample = len(variables_per_class[i]) - int(len(variables_per_class[i]) * test_avg)
    for j in range(int(len(variables_per_class[i]) * test_avg)):
        eq_test.append(variables_per_class[i][min_sample + j])

for i in range(5):
    eq_data = np.array(eq_data)
    np.random.shuffle(eq_data)
    eq_test = np.array(eq_test)
    np.random.shuffle(eq_test)

    X_train = eq_data[:, 1:55]
    y_train = eq_data[:, 55]
    X_test = eq_test[:, 1:55]
    y_test = eq_test[:, 55]
    ##


    ##
    bst = xgb.XGBClassifier()
    bst.fit(X_train, y_train)
    ##


    ##
    y_pred = bst.predict(X_test)
    print(classification_report(y_test, y_pred))
    ##


    ##
    final_predictions = bst.predict(predict[:, 1:])
    with open(r'Output\capicorb_0{}.txt'.format(i), 'w') as file:
        with open(r'Data\Estimar_UH2020.csv', 'r') as read:
            for i in range(len(final_predictions)):
                file.write('{}|{}'.format(read.readline().split('|')[0], dictio_i[final_predictions[i]]))
    ## 

## Evaluación global
last_prediction = {}
for i in range(5):
    with open(r'Output\capicorb_0{}.txt'.format(i), 'r') as file:
        for line in file.readlines():
            line = line.split('|')
            if (last_prediction[line[0]] is None):
                last_prediction[line[0]] = [line[1]]
            else:
                last_prediction[line[0]] = last_prediction[line[0]].append(line[1])

with open(r'Output\capicorb_F.txt', 'w') as file:
    file.write('ID|CLASE')
    with open(r'Data\Estimar_UH2020.csv', 'r') as read:
        for line in read.readlines():
            line = line.split()
            file.write('{}|{}'.format(line[0], max(set(last_prediction[line[0]]), key = last_prediction[line[0]].count)))
## 