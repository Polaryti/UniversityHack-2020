import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.ensemble import RandomForestClassifier

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
X_train, X_test, y_train, y_test = train_test_split(eq_data[:, 1:55], eq_data[:, 55], test_size = 0.12)
###

# scaler = Normalizer().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
# scaler = StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
clf = RandomForestClassifier()
RandomForestClassifier
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))