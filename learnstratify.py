from sklearn.model_selection import train_test_split
import pandas as pd
from datasets_get import get_modelar_data
from not_random_test_generator import dividir_dataset

CLASS = 65

modelar_df = get_modelar_data()
X_modelar = modelar_df.loc[:, modelar_df.columns!=CLASS]
y_modelar = modelar_df.loc[:, CLASS]

train, test = dividir_dataset(modelar_df)
print(test.groupby(66).count())
