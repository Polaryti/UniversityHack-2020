import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

variables_per_class = []
for i in range(7):         
    variables_per_class.append([])
for label in data:
    variables_per_class[int(label[55])].append(label)

eq_data = []
for i in range(90173):
    for j in range(7):
        eq_data.append(variables_per_class[j][i % (len(variables_per_class[j]))])
eq_data = np.array(eq_data)

X_train, X_test, y_train, y_test = train_test_split(eq_data[:, 1:55], eq_data[:, 55], test_size = 0.20)
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
with open('res-capicorb.txt', 'w') as file:
    with open(r'Data\Estimar_UH2020.csv', 'r') as read:
        for i in range(len(final_predictions)):
            file.write('{}|{}'.format(read.readline().split('|')[0], dictio_i[final_predictions[i]]))
## 