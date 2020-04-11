 # coding=utf-8
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split

def dividir_dataset(df, train_percentage=0.8, test_percentage=0.2, randomized=False, residential_perc=14605, industrial_perc=1473, public_perc=1433, retail_perc=897, office_perc=648, other_perc=1361, agriculture_perc=229):
    np.random.seed(7)
    if train_percentage + test_percentage != 1:
        print("El dataset debe emplearse entero, los porcentajes de train y test deben sumar 1.")
    elif randomized == True:
        train,test = train_test_split(df, test_size=0.2)
        return train, test
    else:
        percentages_list = [residential_perc, industrial_perc, public_perc, retail_perc, office_perc, other_perc, agriculture_perc]
        df_list = []
        for i in range(7):
            current_df = df.loc[df[66] == i]
            remove_num = percentages_list[i]
            df_list.append(current_df.sample(n=remove_num, replace=True))
        for i in range(1,7):
            df_list[0] = df_list[0].append(df_list[i], ignore_index=True)
        df = pd.concat([df, df_list[0]]).drop_duplicates(keep=False)
        #Devuelve una tupla, en 0 el train y en 1 el test.
        return df.sample(frac=1), df_list[0].sample(frac=1)