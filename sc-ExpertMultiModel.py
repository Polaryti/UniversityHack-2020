import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import LocallyLinearEmbedding, SpectralEmbedding
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import MeanShift

class MultiModel:
    data = []
    data_predict = []
    variables_per_class = []
    glb_predictions = {}
    nmb_predictions = 8

    def agriculture_model(self):
        lcl_data = []
        for i in range(338):
            for j in range(7):
                lcl_data.append(self.variables_per_class[j][i])
        lcl_data = np.array(lcl_data)

        clf = RandomForestClassifier()

        self.data_predict = np.array(self.data_predict[:, 1:55])
        for _ in range(self.nmb_predictions):
            X_train, X_test, y_train, y_test = train_test_split(
                lcl_data[:, 1:55], lcl_data[:, 55], test_size=0.15)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print(classification_report(y_test, y_pred))
            self.global_predict(6, clf.predict(self.data_predict))


    def global_predict(self, class_number, predictions):
        for i in range(len(predictions)):
            if predictions[i] == class_number:
                if int(self.data_predict[i][0]) not in self.glb_predictions:
                    self.glb_predictions[int(self.data_predict[i][0])] = [class_number]
                else:
                    self.glb_predictions[int(self.data_predict[i][0])].append(class_number)


    def gen_data(self):
        # preprocessing()
        self.data = np.genfromtxt(r'Data\Modelar_UH2020.csv', delimiter=',')
        self.data_predict = np.genfromtxt(r'Data\Estimar_UH2020.csv', delimiter='|')

        for i in range(7):
            self.variables_per_class.append([])
        for label in self.data:
            self.variables_per_class[int(label[55])].append(label)


    def preprocessing(self):
        pass

    def __init__(self):
        self.gen_data()
        self.agriculture_model()
        for ele in self.glb_predictions:
            print('{} -> {}'.format(ele, self.glb_predictions[ele]))


if __name__ == "__main__":
    model = MultiModel()