import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer

test_avg = 0.2

avg_prediction = {}
train_prediction = {}

dictio_i = {0: 'RESIDENTIAL\n',
    1: 'INDUSTRIAL\n',
    2: 'PUBLIC\n',
    3: 'OFFICE\n',
    4: 'OTHER\n',
    5: 'RETAIL\n',
    6: 'AGRICULTURE\n'}

def most_frequent(List): 
    return max(set(List), key = List.count) 

##
data = np.genfromtxt(r'Data\Modelar_UH2020.csv', delimiter = ',')
predict = np.genfromtxt(r'Data\Estimar_UH2020.csv', delimiter='|')

for iteration in range(10):
    np.random.shuffle(data)
    # variables_per_class = []
    # for i in range(7):         
    #     variables_per_class.append([])
    # for label in data:
    #     variables_per_class[int(label[55])].append(label)

    ## Data normalizada al por menor
    # eq_data_menor = []
    # for i in range(338):
    #     for j in range(7):
    #         if j is 6:
    #             eq_data_menor.append(variables_per_class[j][i % 338])
    #         else:
    #             eq_data_menor.append(variables_per_class[j][i])
    # eq_data_menor = np.array(eq_data_menor)
    # for i in range(90000):
    #     eq_data_menor.append(variables_per_class[0][i])
    # eq_data_menor += variables_per_class[1]
    # eq_data_menor += variables_per_class[2]
    # eq_data_menor += variables_per_class[3]
    # eq_data_menor += variables_per_class[4]
    # eq_data_menor += variables_per_class[5]
    # eq_data_menor += variables_per_class[6]
    # eq_data_menor = np.array(eq_data_menor)
    
    X_train_menor, X_test_menor, y_train_menor, y_test_menor = train_test_split(data[:, 1:55], data[:, 55], test_size = test_avg)
    normalizer = Normalizer().fit(X_train_menor)
    X_train_menor = normalizer.transform(X_train_menor)
    X_test_menor = normalizer.transform(X_test_menor)
    ##


    ##
    bst = xgb.XGBClassifier()
    bst.fit(X_train_menor, y_train_menor)
    y_pred = bst.predict(X_test_menor)
    # print(confusion_matrix(y_test_menor, y_pred))
    print(classification_report(y_test_menor, y_pred))
    print('{}_XGB_m - {}'.format(iteration, accuracy_score(y_test_menor, y_pred)))
    final_predictions = bst.predict(predict[:, 1:])
    with open(r'Output\torrent\torrent_0{}_XGB_m.txt'.format(iteration), 'w') as file:
        with open(r'Data\Estimar_UH2020.csv', 'r') as read:
            for i in range(len(final_predictions)):
                line = read.readline().split('|')
                file.write('{}|{}'.format(line[0], dictio_i[final_predictions[i]]))
                if line[0] not in avg_prediction:
                    avg_prediction[line[0]] = [int(final_predictions[i])]
                else:
                    avg_prediction[line[0]].append(int(final_predictions[i]))
    # final_predictions = bst.predict(data[:, 1:55])
    # for ite in range(len(data)):
    #     if data[ite][0] not in train_prediction:
    #         train_prediction[data[ite][0]] = [final_predictions[ite]]   #[int(bst.predict(np.array([sample[1:55]])))]
    #     else:
    #         train_prediction[data[ite][0]].append(final_predictions[ite])
    ##
    ##
    #bst = RandomForestClassifier(n_estimators = 400, min_samples_split = 5, min_samples_leaf = 1, max_features = 'auto', max_depth = 60, bootstrap = True)
    bst = RandomForestClassifier()
    bst.fit(X_train_menor, y_train_menor)
    y_pred = bst.predict(X_test_menor)
    # print(confusion_matrix(y_test_menor, y_pred))
    print(classification_report(y_test_menor, y_pred))
    print('{}_rndf_m - {}'.format(iteration, accuracy_score(y_test_menor, y_pred)))
    final_predictions = bst.predict(predict[:, 1:])
    with open(r'Output\torrent\torrent_0{}_rndf_m.txt'.format(iteration), 'w') as file:
        with open(r'Data\Estimar_UH2020.csv', 'r') as read:
            for i in range(len(final_predictions)):
                line = read.readline().split('|')
                file.write('{}|{}'.format(line[0], dictio_i[final_predictions[i]]))
                if (line[0] not in avg_prediction):
                    avg_prediction[line[0]] = [int(final_predictions[i])]
                else:
                    avg_prediction[line[0]].append(int(final_predictions[i]))
    final_predictions = bst.predict(data[:, 1:55])
    # for ite in range(len(data)):
    #     if data[ite][0] not in train_prediction:
    #         train_prediction[data[ite][0]] = [final_predictions[ite]]   #[int(bst.predict(np.array([sample[1:55]])))]
    #     else:
    #         train_prediction[data[ite][0]].append(final_predictions[ite])
    ##

# for iteration in range(4):
#     np.random.shuffle(data)
#     variables_per_class = []
#     for i in range(7):         
#         variables_per_class.append([])
#     for label in data:
#         variables_per_class[int(label[55])].append(label)

    ## Data normalizado al por mayor
    # eq_data_major = []
    # for i in range(90173):
    #     for j in range(7):
    #         max_sample = len(variables_per_class[j]) - int(len(variables_per_class[j]) * test_avg)
    #         eq_data_major.append(variables_per_class[j][i % max_sample])

    # eq_test_major = []
    # for i in range(7):
    #     min_sample = len(variables_per_class[i]) - int(len(variables_per_class[i]) * test_avg)
    #     for j in range(int(len(variables_per_class[i]) * test_avg)):
    #         eq_test_major.append(variables_per_class[i][min_sample + j])

    # eq_data_major = np.array(eq_data_major)
    # np.random.shuffle(eq_data_major)
    # eq_test_major = np.array(eq_test_major)
    # np.random.shuffle(eq_test_major)

    # X_train_major = eq_data_major[:, 1:55]
    # y_train_major = eq_data_major[:, 55]
    # X_test_major = eq_test_major[:, 1:55]
    # y_test_major = eq_test_major[:, 55]

    # ##
    # bst = xgb.XGBClassifier()
    # bst.fit(X_train_major, y_train_major)
    # y_pred = bst.predict(X_test_major)
    # # print(confusion_matrix(y_test_major, y_pred))
    # print(classification_report(y_test_major, y_pred))
    # print('{}_XGB_M - {}'.format(iteration, accuracy_score(y_test_major, y_pred)))
    # final_predictions = bst.predict(predict[:, 1:])
    # with open(r'Output\torrent\torrent_0{}_XGB_M.txt'.format(iteration), 'w') as file:
    #     with open(r'Data\Estimar_UH2020.csv', 'r') as read:
    #         for i in range(len(final_predictions)):
    #             line = read.readline().split('|')
    #             file.write('{}|{}'.format(line[0], dictio_i[final_predictions[i]]))
    #             if (line[0] not in avg_prediction):
    #                 avg_prediction[line[0]] = [int(final_predictions[i])]
    #             else:
    #                 avg_prediction[line[0]].append(int(final_predictions[i]))
    # ##
    # ##
    # bst = RandomForestClassifier(n_estimators = 400, min_samples_split = 5, min_samples_leaf = 1, max_features = 'auto', max_depth = 60, bootstrap = True)
    # bst.fit(X_train_major, y_train_major)
    # y_pred = bst.predict(X_test_major)
    # # print(confusion_matrix(y_test_major, y_pred))
    # # print(classification_report(y_test_major, y_pred))
    # print('{}_rndf_M - {}'.format(iteration, accuracy_score(y_test_major, y_pred)))
    # final_predictions = bst.predict(predict[:, 1:])
    # with open(r'Output\torrent\torrent_0{}_rndf_M.txt'.format(iteration), 'w') as file:
    #     with open(r'Data\Estimar_UH2020.csv', 'r') as read:
    #         for i in range(len(final_predictions)):
    #             line = read.readline().split('|')
    #             file.write('{}|{}'.format(line[0], dictio_i[final_predictions[i]]))
    #             if (line[0] not in avg_prediction):
    #                 avg_prediction[line[0]] = [int(final_predictions[i])]
    #             else:
    #                 avg_prediction[line[0]].append(int(final_predictions[i]))
    # ##
    

## Evaluación global
with open(r'Output\torrent\torrent_DEBUG.txt', 'w') as debug:
    for item in avg_prediction.items():
        debug.write('{}\n'.format(str(item)))
with open(r'Data\Estimar_UH2020.csv', 'r') as read:
    with open(r'Output\torrent\torrent_FINAL.txt', 'w') as file:
        for line in read.readlines():
            line = line.split('|')
            file.write('{}|{}'.format(line[0], dictio_i[most_frequent(avg_prediction[line[0]])]))

global_avg = []
real_pred = []
for sample in data:
    if sample[0] in train_prediction:
        global_avg.append(train_prediction[most_frequent(avg_prediction[sample[0]])])
        real_pred.append(sample[55])
print(classification_report(real_pred, global_avg))
## 