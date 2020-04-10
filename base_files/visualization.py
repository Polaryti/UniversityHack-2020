import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import random
import XGB_RandomForestBasic_bunyol

categories_list = XGB_RandomForestBasic_bunyol.get_categories_list()
modelar_df = XGB_RandomForestBasic_bunyol.get_modelar_data()
estimar_df = XGB_RandomForestBasic_bunyol.get_estimar_data()

def hist_decomposition():
    count_df = pd.DataFrame(columns = categories_list)
    for i in range(len(categories_list)):
        count_df[categories_list[i]] = [pd.DataFrame(modelar_df.loc[modelar_df[54] == i]).shape[0]]
    count_df.plot(kind='bar', title='Número de elementos de cada clase')


def hist_over_and_undersampling(y_res):
    y_res.value_counts().plot(kind='bar', title='Número de muestras por clase')


def pca_general(X, y, d2=True, d3=True, pie_evr=True):
    if d2 == True:
        pca_2d(X, y)
    if d3 == True:
        pca_3d(X, y)

def pca_2d(X, y, n_components=2, pie_evr=True):
    pca = PCA(n_components=n_components)
    pca_transform = pca.fit_transform(X)
    X['pca_first'] = pca_transform[:,0]
    X['pca_second'] = pca_transform[:,1]
    if pie_evr:
        pie_explained_variance_ratio(pca.explained_variance_ratio_)
    plt.figure(figsize=(14,10))
    sns.scatterplot(
    x="pca_first", y="pca_second",
    palette=sns.color_palette("hls", 7),
    hue='Class',
    data=X,
    legend="full",
    alpha=0.3
    )
    plt.show()


def pca_3d(X, y, n_components=3, pie_evr=True):
    pca = PCA(n_components=n_components)
    pca_transform = pca.fit_transform(X)
    X['pca_first'] = pca_transform[:,0]
    X['pca_second'] = pca_transform[:,1]
    X['pca_third'] = pca_transform[:,2]
    if pie_evr:
        pie_explained_variance_ratio(pca.explained_variance_ratio_)
    ax = plt.figure(figsize=(14,10)).gca(projection='3d')
    ax.scatter(
    xs=X["pca_first"],
    ys=X["pca_second"],
    zs=X["pca_third"],
    c=X['Class'],
    cmap='tab10'
    )
    ax.set_xlabel('pca_first')
    ax.set_ylabel('pca_second')
    ax.set_zlabel('pca_third')
    plt.show()


def tsne(X, y, perplexity):
    for perp in perplexity:
        tsne = TSNE(n_components=2, verbose=0, perplexity=perp, n_iter=400)
        features = list(X.columns.values)
        tsne_results = tsne.fit_transform(X[features].values)
        X['tsne-2d-first'] = tsne_results[:,0]
        X['tsne-2d-second'] = tsne_results[:,1]
        print('Perplexity = ',perp)
        plt.figure(figsize=(14,10))
        sns.scatterplot(
        x="tsne-2d-first", y="tsne-2d-first",
        hue="y",
        palette=sns.color_palette("hls", 7),
        data=X,
        legend="full",
        alpha=0.3
        )
        plt.show()


def pie_explained_variance_ratio(explained_variance_ratio):
    evr_df = pd.DataFrame(explained_variance_ratio)
    evr_df.plot.pie(y=0, figsize=(5,5))
    i = 0
    index = 0
    for item in explained_variance_ratio:
        i+=item
        print('La componente principal nº {} es responsable del {} por ciento de la varianza.'.format(index, item*100))
        index += 1
    i*=100
    print('Captura el {} por ciento de la información total'.format(i))
