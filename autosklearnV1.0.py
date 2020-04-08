from mlbox.preprocessing import Reader
from mlbox.preprocessing import Drift_thresholder
from mlbox.optimisation import Optimiser
from mlbox.prediction import Predictor
import random


data = []
with open(r'Data\Modelar_UH2020.txt') as read_file:
    _ = read_file.readline()
    for line in read_file.readlines():
        data.append(line)

random.shuffle(data)

with open(r'Data\askl_train.csv', 'w') as write_file:
    write_file.write('X|Y|Q_R_4_0_0|Q_R_4_0_1|Q_R_4_0_2|Q_R_4_0_3|Q_R_4_0_4|Q_R_4_0_5|Q_R_4_0_6|Q_R_4_0_7|Q_R_4_0_8|Q_R_4_0_9|Q_R_4_1_0|Q_G_3_0_0|Q_G_3_0_1|Q_G_3_0_2|Q_G_3_0_3|Q_G_3_0_4|Q_G_3_0_5|Q_G_3_0_6|Q_G_3_0_7|Q_G_3_0_8|Q_G_3_0_9|Q_G_3_1_0|Q_B_2_0_0|Q_B_2_0_1|Q_B_2_0_2|Q_B_2_0_3|Q_B_2_0_4|Q_B_2_0_5|Q_B_2_0_6|Q_B_2_0_7|Q_B_2_0_8|Q_B_2_0_9|Q_B_2_1_0|Q_NIR_8_0_0|Q_NIR_8_0_1|Q_NIR_8_0_2|Q_NIR_8_0_3|Q_NIR_8_0_4|Q_NIR_8_0_5|Q_NIR_8_0_6|Q_NIR_8_0_7|Q_NIR_8_0_8|Q_NIR_8_0_9|Q_NIR_8_1_0|AREA|GEOM_R1|GEOM_R2|GEOM_R3|GEOM_R4|CONTRUCTIONYEAR|MAXBUILDINGFLOOR|CADASTRALQUALITYID|CLASE\n')
    for line in data[:80000]:
        line = line.split('|')
        line = line[1:]
        write_file.write('{}'.format('|'.join(line)))

with open(r'Data\askl_test.csv', 'w') as write_file_a:
    write_file_a.write('X|Y|Q_R_4_0_0|Q_R_4_0_1|Q_R_4_0_2|Q_R_4_0_3|Q_R_4_0_4|Q_R_4_0_5|Q_R_4_0_6|Q_R_4_0_7|Q_R_4_0_8|Q_R_4_0_9|Q_R_4_1_0|Q_G_3_0_0|Q_G_3_0_1|Q_G_3_0_2|Q_G_3_0_3|Q_G_3_0_4|Q_G_3_0_5|Q_G_3_0_6|Q_G_3_0_7|Q_G_3_0_8|Q_G_3_0_9|Q_G_3_1_0|Q_B_2_0_0|Q_B_2_0_1|Q_B_2_0_2|Q_B_2_0_3|Q_B_2_0_4|Q_B_2_0_5|Q_B_2_0_6|Q_B_2_0_7|Q_B_2_0_8|Q_B_2_0_9|Q_B_2_1_0|Q_NIR_8_0_0|Q_NIR_8_0_1|Q_NIR_8_0_2|Q_NIR_8_0_3|Q_NIR_8_0_4|Q_NIR_8_0_5|Q_NIR_8_0_6|Q_NIR_8_0_7|Q_NIR_8_0_8|Q_NIR_8_0_9|Q_NIR_8_1_0|AREA|GEOM_R1|GEOM_R2|GEOM_R3|GEOM_R4|CONTRUCTIONYEAR|MAXBUILDINGFLOOR|CADASTRALQUALITYID\n')
    for line in data[80000:]:
        line = line.split('|')
        last = len(line) - 1
        line = line[1:last]
        write_file_a.write('{}\n'.format('|'.join(line)))       


# Paths to the train set and the test set.
paths = [r'Data\askl_train.csv', r'Data\askl_test.csv']
# Name of the feature to predict.
# This columns should only be present in the train set.
target_name = "CLASE"

# Reading and cleaning all files
# Declare a reader for csv files
rd = Reader(sep='|')
# Return a dictionnary containing three entries
# dict["train"] contains training samples withtout target columns
# dict["test"] contains testing elements withtout target columns
# dict["target"] contains target columns for training samples.
data = rd.train_test_split(paths, target_name)

dft = Drift_thresholder()
data = dft.fit_transform(data)

# Tuning
# Declare an optimiser. Scoring possibilities for classification lie in :
# {"accuracy", "roc_auc", "f1", "neg_log_loss", "precision", "recall"}
opt = Optimiser(scoring='accuracy', n_folds=3)
opt.evaluate(None, data)

# Space of hyperparameters
# The keys must respect the following syntax : "enc__param".
#   "enc" = "ne" for na encoder
#   "enc" = "ce" for categorical encoder
#   "enc" = "fs" for feature selector [OPTIONAL]
#   "enc" = "stck"+str(i) to add layer n°i of meta-features [OPTIONAL]
#   "enc" = "est" for the final estimator
#   "param" : a correct associated parameter for each step.
#   Ex: "max_depth" for "enc"="est", ...
# The values must respect the syntax: {"search":strategy,"space":list}
#   "strategy" = "choice" or "uniform". Default = "choice"
#   list : a list of values to be tested if strategy="choice".
#   Else, list = [value_min, value_max].
# Available strategies for ne_numerical_strategy are either an integer, a float
#   or in {'mean', 'median', "most_frequent"}
# Available strategies for ce_strategy are:
#   {"label_encoding", "dummification", "random_projection", entity_embedding"}
space = {'ne__numerical_strategy': {"search": "choice", "space": [0]},
         'ce__strategy': {"search": "choice",
                          "space": ["label_encoding",
                                    "random_projection",
                                    "entity_embedding"]},
         'fs__threshold': {"search": "uniform",
                           "space": [0.01, 0.3]},
         'est__max_depth': {"search": "choice",
                            "space": [3, 4, 5, 6, 7]}

         }

# Optimises hyper-parameters of the whole Pipeline with a given scoring
# function. Algorithm used to optimize : Tree Parzen Estimator.
#
# IMPORTANT : Try to avoid dependent parameters and to set one feature
# selection strategy and one estimator strategy at a time.
best = opt.optimise(space, data, 15)

# Make prediction and save the results in save folder.
prd = Predictor()
prd.fit_predict(best, data)