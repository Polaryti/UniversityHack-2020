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

    """
    print(list(X_modelar.columns.values))
    print(list(X_estimar.columns.values))
    print(list(y_modelar.columns.values))
    """

    classifier = xgb.XGBClassifier()
    ovsr = OneVsRestClassifier(classifier,n_jobs=-1).fit(X_modelar,y_modelar)
    pred_estimar = ovsr.predict_proba(X_estimar)

    offset = X_modelar.shape[0]
    classes = get_categories_list()
    col_names = []

    for i in range(7):
        col_names.append('coords_' + classes[i])

    cont = [] 

    for i in range(X_est_mod.shape[0]):
        
        indices = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        
        neigh_dist, neigh_indices = spatialTree.query([[coords.iloc[i,0],coords.iloc[i,1]]],k=K)
        
        for j in range(1,K):
            # Para cada vecino sumamos 1 a la variable contexto de la clase de la finca 
            # O en caso de que se encuentre en X_estimar sumamos las probabilidades
            if neigh_indices[0][j] < offset : 
                indices[int(y_modelar.loc[neigh_indices[0][j], 'CLASS'])] += 1
            else:
                
                indices = np.add(indices, pred_estimar[neigh_indices[0][j]-offset,:])

        cont.append(indices)# Sin softmax
        #cont.append(softmax(np.array(indices))) #Con softmax

    indexes_est = []
    for i in range(X_estimar.shape[0]):
        indexes_est.append(i)

    context  = pd.DataFrame(data=cont,columns=col_names)
    context_modelar = context.loc[:offset-1]
    context_estimar = context.loc[offset:]
    #context_estimar.reset_index()
    context_estimar.index = range(5618)

    #context.drop('coords_RESIDENTIAL',axis=1,inplace=True) #PROBAR CON Y SIN

    for column in col_names:
        X_modelar[column] = context_modelar[column]
        X_estimar[column] = context_estimar[column]


    return X_modelar.values, X_estimar.values


coordinates_fe(getX(get_modelar_data_ids()), getY(get_modelar_data()), get_estimar_data())