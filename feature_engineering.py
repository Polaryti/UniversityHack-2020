import pandas as pd
import numpy as np
from datasets_get import get_modelar_data, reduce_dimension_modelar
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.multiclass import OneVsRestClassifier
from datasets_get import get_modelar_data, get_estimar_data, getX, getY, get_modelar_data_ids, get_categories_list
from datasets_get import reduce_colors
from scipy.spatial import cKDTree
from scipy.special import softmax

columns = 'X|Y|Q_R_4_0_0|Q_R_4_0_1|Q_R_4_0_2|Q_R_4_0_3|Q_R_4_0_4|Q_R_4_0_5|Q_R_4_0_6|Q_R_4_0_7|Q_R_4_0_8|Q_R_4_0_9|Q_R_4_1_0|Q_G_3_0_0|Q_G_3_0_1|Q_G_3_0_2|Q_G_3_0_3|Q_G_3_0_4|Q_G_3_0_5|Q_G_3_0_6|Q_G_3_0_7|Q_G_3_0_8|Q_G_3_0_9|Q_G_3_1_0|Q_B_2_0_0|Q_B_2_0_1|Q_B_2_0_2|Q_B_2_0_3|Q_B_2_0_4|Q_B_2_0_5|Q_B_2_0_6|Q_B_2_0_7|Q_B_2_0_8|Q_B_2_0_9|Q_B_2_1_0|Q_NIR_8_0_0|Q_NIR_8_0_1|Q_NIR_8_0_2|Q_NIR_8_0_3|Q_NIR_8_0_4|Q_NIR_8_0_5|Q_NIR_8_0_6|Q_NIR_8_0_7|Q_NIR_8_0_8|Q_NIR_8_0_9|Q_NIR_8_1_0|AREA|GEOM_R1|GEOM_R2|GEOM_R3|GEOM_R4|CONTRUCTIONYEAR|MAXBUILDINGFLOOR|CADASTRALQUALITYID_00|CADASTRALQUALITYID_01|CADASTRALQUALITYID_02|CADASTRALQUALITYID_03|CADASTRALQUALITYID_04|CADASTRALQUALITYID_05|CADASTRALQUALITYID_06|CADASTRALQUALITYID_07|CADASTRALQUALITYID_08|CADASTRALQUALITYID_09|CADASTRALQUALITYID_10|CADASTRALQUALITYID_11|CADASTRALQUALITYID_12|CLASS'.split('|')
columns_dict = {}
for i in range(len(columns)):
    columns_dict[i] = columns[i]

def coordinates_fe(X_modelar, y_modelar, X_estimar):
    print(y_modelar)
    est_ID = X_estimar[0]
    X_full = pd.concat([X_modelar, X_estimar], sort=False)
    coords = X_full[[1, 2]].rename(columns={1:'X', 2:'Y'})

    spatialTree = cKDTree(np.c_[coords.X.ravel(),coords.Y.ravel()])

    X_full.drop([0,1,2],inplace=True,axis=1)
    X_full = reduce_colors(X_full)

    X_estimar.drop([0,1,2],inplace=True,axis=1)
    X_estimar = reduce_colors(X_estimar)

    X_modelar.drop([0,1,2],inplace=True,axis=1)
    X_modelar = reduce_colors(X_modelar)

    model = xgb.XGBClassifier()
    ovsr = OneVsRestClassifier(model,n_jobs=-1).fit(X_modelar.values,y_modelar.values)
    pred_estimar = ovsr.predict_proba(X_estimar.values)

    K = 4 

    offset = X_modelar.shape[0]
    classes = get_categories_list()

    col_names = []
    y_map = {'RESIDENTIAL':5,'INDUSTRIAL':1,'PUBLIC':4,'RETAIL':6,'OFFICE':2,'OTHER':3,'AGRICULTURE':0}

    for i in range(7):
        col_names.append('coords_' + classes[i])

    tmp_concat = [] # Por eficiencia, el coste de concatenacion de un df en el for es cuadratico

    for i in range(X_full.shape[0]):
        
        #Inicializamos las variables contexto
        tmp = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        
        #Obtenemos los vecinos
        vecinos_dist, vecinos_indices = spatialTree.query([[coords.iloc[i,0],coords.iloc[i,1]]],k=K)
        
        for j in range(1,K):
            # Para cada vecino sumamos 1 a la variable contexto de la clase de la finca 
            # O en caso de que se encuentre en X_estimar sumamos las probabilidades
            if vecinos_indices[0][j] < offset : 
                tmp[y_map[y_modelar[j]]] = tmp[y_map[y_modelar[j]]] + 1       
            else:
                tmp = np.add(tmp, pred_estimar[vecinos_indices[0][j]-offset,:])

        tmp_concat.append(tmp)# Sin softmax
        #tmp_concat.append(softmax(np.array(tmp))) #Con softmax

    contexto  = pd.DataFrame(tmp_concat,columns=col_names)
    print(contexto.shape)
    print(X_full.shape)
    contexto.head(20)
    contexto.drop('coords_RESIDENTIAL',axis=1,inplace=True) #PROBAR CON Y SIN
    X_modelar2 = pd.concat([X_modelar,contexto[:X_modelar.shape[0]].reindex(X_modelar.index)], axis = 1, sort = False)
    X_estimar2 = pd.concat([X_estimar, contexto[X_modelar.shape[0]:].reindex(X_estimar.index)], axis = 1, sort = False)

    return X_modelar2, X_estimar2



coordinates_fe(getX(get_modelar_data_ids()), getY(get_modelar_data()), get_estimar_data())
