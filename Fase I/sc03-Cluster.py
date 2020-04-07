import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, adjusted_rand_score
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.cluster import KMeans

data = np.genfromtxt(r'Data\Modelar_UH2020.csv', delimiter = ',') # Preprocesamiento previo (codificación categorias, '|' -> ','...)
X_train, X_test, y_train, y_test = train_test_split(data[:, 1:55], data[:, 55], test_size = 0.01)

# scaler = Normalizer().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
# scaler = StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

kmeans = KMeans(n_clusters = 7, random_state = 0)
kmeans.fit(X_train)

y_pred = kmeans.predict(X_test)
adjusted_rand_score(y_train, y_pred)