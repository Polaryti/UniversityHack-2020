import pandas as pd
import numpy as np
from datasets_get import get_modelar_data, reduce_dimension_modelar
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.multiclass import OneVsRestClassifier
from datasets_get import get_modelar_data, get_estimar_data, getX, getY, get_modelar_data_ids, get_categories_list, reduce_geometry_average
from datasets_get import reduce_colors
from scipy.spatial import cKDTree
from scipy.special import softmax


def coordinates_fe(X_modelar, y_modelar, X_estimar, K=4):
    est_IDs = X_estimar[0]
    X_est_mod = pd.concat([X_modelar, X_estimar], sort=False)
    coords = X_est_mod[[1, 2]].rename(columns={1:'X', 2:'Y'})

    spatialTree = cKDTree(np.c_[coords.X.ravel(),coords.Y.ravel()])

    X_est_mod.drop([0,1,2],inplace=True,axis=1)
    X_est_mod = reduce_colors(X_est_mod)

    X_estimar.drop([0,1,2],inplace=True,axis=1)
    X_estimar = reduce_colors(X_estimar)

    X_modelar.drop([0,1,2],inplace=True,axis=1)
    X_modelar = reduce_colors(X_modelar)

    print(list(X_modelar.columns.values))
    print(list(X_estimar.columns.values))
    print(list(y_modelar.columns.values))
    classifier = xgb.XGBClassifier()
    ovsr = OneVsRestClassifier(classifier,n_jobs=-1).fit(X_modelar.values,y_modelar.values)
    pred_estimar = ovsr.predict_proba(X_estimar.values)

    offset = X_modelar.shape[0]
    classes = get_categories_list()

    col_names = []

    for i in range(7):
        col_names.append('coords_' + classes[i])

    cont = [] 

    for i in range(X_est_mod.shape[0]):
        
        indices = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        
        neigh_dist, neigh_indices = spatialTree.query([[coords.iloc[i,0],coords.iloc[i,1]]],k=K)
        
        if i == 0:
            print(neigh_indices)
        
        for j in range(1,K):
            # Para cada vecino sumamos 1 a la variable contexto de la clase de la finca 
            # O en caso de que se encuentre en X_estimar sumamos las probabilidades
            if neigh_indices[0][j] < offset : 
                indices[int(y_modelar.loc[neigh_indices[0][j], 'CLASS'])] += 1
            else:
                indices = np.add(indices, pred_estimar[neigh_indices[0][j]-offset,:])

        #cont.append(indices)# Sin softmax
        cont.append(softmax(np.array(indices))) #Con softmax

    context  = pd.DataFrame(cont,columns=col_names)
    print(context)
    #context.drop('coords_RESIDENTIAL',axis=1,inplace=True) #PROBAR CON Y SIN
    X_modelar_fe = pd.concat([X_modelar,context[:X_modelar.shape[0]].reindex(X_modelar.index)], axis = 1, sort = False)
    X_estimar_fe = pd.concat([X_estimar, context[X_modelar.shape[0]:].reindex(X_estimar.index)], axis = 1, sort = False)

    print(X_modelar_fe)
    print(X_estimar_fe)
    return X_modelar_fe.values, X_estimar_fe.values


coordinates_fe(getX(get_modelar_data_ids()), getY(get_modelar_data()), get_estimar_data())