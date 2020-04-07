import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# read in data
data = np.genfromtxt(r'Data\Modelar_UH2020.csv', delimiter = ',')
predict = np.genfromtxt(r'Data\Estimar_UH2020.csv', delimiter='|')

dictio_i = {0: 'RESIDENTIAL\n',
    1: 'INDUSTRIAL\n',
    2: 'PUBLIC\n',
    3: 'OFFICE\n',
    4: 'OTHER\n',
    5: 'RETAIL\n',
    6: 'AGRICULTURE\n'}

X_train, X_test, y_train, y_test = train_test_split(data[:, 1:55], data[:, 55], test_size = 0.20)

# specify parameters via map
param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic'}
num_round = 2
bst = xgb.XGBClassifier()
bst.fit(X_train, y_train)

# make prediction
y_pred = bst.predict(X_test)
print(y_pred)
print(classification_report(y_test, y_pred))

final_predictions = bst.predict(predict[:, 1:])
with open('res-xeraco.txt', 'w') as file:
    with open(r'Data\Estimar_UH2020.csv', 'r') as read:
        for i in range(len(final_predictions)):
            file.write('{} - {}'.format(read.readline().split('|')[0], dictio_i[final_predictions[i]]))