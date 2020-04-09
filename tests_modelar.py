 # coding=utf-8
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
import XGB_RandomForestBasic_bunyol


def dividir_dataset(df, train_percentage=0.8, test_percentage=0.2, randomized=False, residential_perc=0.1, industrial_perc=0.15, public_perc=0.15, retail_perc=0.1, office_perc=0.15, other_perc=0.15, agriculture_perc=0.2):
    np.random.seed(7)
    if train_percentage + test_percentage != 1:
        print("El dataset debe emplearse entero, los porcentajes de train y test deben sumar 1.")
    elif residential_perc + industrial_perc + public_perc + retail_perc + office_perc + other_perc + agriculture_perc != 1:
        print("Los porcentajes de cada clase deben sumar 1.")
    elif randomized == True:
        train,test = train_test_split(df, test_size=0.2)
        return train, test
    else:
        total_num = df.shape[0]
        percentages_list = [residential_perc, industrial_perc, public_perc, retail_perc, office_perc, other_perc, agriculture_perc]
        df_list = []
        for i in range(7):
            current_df = df.loc[df[54] == i]
            remove_num = int(math.ceil(total_num * test_percentage * percentages_list[i]))
            df_list.append(current_df.sample(n=remove_num, replace=True))
        for i in range(1,7):
            df_list[0] = df_list[0].append(df_list[i], ignore_index=True)
        df = pd.concat([df, df_list[0]]).drop_duplicates(keep=False)
        #Devuelve una tupla, en 0 el train y en 1 el test.
        return df.sample(frac=1), df_list[0].sample(frac=1)
