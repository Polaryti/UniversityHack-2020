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
from mpl_toolkits.mplot3d import Axes3D
from datasets_get import get_modelar_data, get_estimar_data, get_categories_list, getX, getY, get_mod_data_original, get_categorical_decoder_class

def hist_decomposition():
    categories_list = get_categories_list()
    modelar_df = get_modelar_data()
    count_df = pd.DataFrame(columns = categories_list)
    for i in range(len(categories_list)):
        count_df[categories_list[i]] = [pd.DataFrame(modelar_df.loc[modelar_df['CLASS'] == i]).shape[0]]
    count_df.plot(kind='bar', title='Número de muestras de cada clase')
    plt.show()


def hist_over_and_undersampling(y_res):
    y_res.value_counts().plot(kind='bar', title='Número de muestras por clase')


def pca_general(X, y, d2=True, d3=True):
    if d2 == True:
        pca_2d(X, y)
    if d3 == True:
        pca_3d(X, y)


def pca_2d(X, y, n_components=2):
    pca = PCA(n_components=n_components)
    pca_transform = pca.fit_transform(X)
    X['pca_first'] = pca_transform[:,0]
    X['pca_second'] = pca_transform[:,1]
    X['CLASS'] = y
    decoder = get_categorical_decoder_class()
    X = X.replace({'CLASS' : decoder})
    print(pca.explained_variance_ratio_)
    print(X)
    plt.figure(figsize=(15,10))
    sns.scatterplot(
    x="pca_first", y="pca_second",
    palette=sns.color_palette("hls", 7),
    hue='CLASS',
    data=X
    )
    plt.show()


def pca_3d(X, y, n_components=3):
    pca = PCA(n_components=n_components)
    pca_transform = pca.fit_transform(X)
    X['pca_first'] = pca_transform[:,0]
    X['pca_second'] = pca_transform[:,1]
    X['pca_third'] = pca_transform[:,2]
    X['CLASS'] = y
    print(pca.explained_variance_ratio_)
    ax = plt.figure(figsize=(15,10)).gca(projection='3d')
    scatter = ax.scatter(
    xs=X["pca_first"],
    ys=X["pca_second"],
    zs=X["pca_third"],
    c=X['CLASS'],
    cmap='tab10',
    alpha=0.3
    )
    ax.set_xlabel('pca_first')
    ax.set_ylabel('pca_second')
    ax.set_zlabel('pca_third')
    plt.show()


def tsne(X, y, perplexity, comp):
    for perp in perplexity:
        tsne = TSNE(n_components=comp, verbose=0, perplexity=perp, n_iter=250)
        features = list(X.columns.values)
        tsne_results = tsne.fit_transform(X[features].values)
        X['tsne-2d-first'] = tsne_results[:,0]
        X['tsne-2d-second'] = tsne_results[:,1]
        X['CLASS'] = y
        decoder = get_categorical_decoder_class()
        X = X.replace({'CLASS' : decoder})
        print('Perplexity = ',perp)
        plt.figure(figsize=(14,10))
        sns.scatterplot(
        x="tsne-2d-first", y="tsne-2d-second",
        hue='CLASS',
        palette=sns.color_palette("hls", 7),
        data=X,
        legend="full",
        alpha=0.3
        )
        plt.show()


def violin_plot():
    df = get_mod_data_original()
    decoder = get_categorical_decoder_class()
    df = df.replace({'CLASS' : decoder})
    df_columns = list(df.columns.values)
    del df_columns[-1]
    for col in df_columns:
        ax = plt.figure(figsize=(14,7))
        ax = sns.violinplot(x='CLASS', y=df[col], data=df)
        plt.show


def violin_plot_kdtree(df, y):
    df['CLASS'] = y
    decoder = get_categorical_decoder_class()
    df = df.replace({'CLASS' : decoder})
    df_columns = list(df.columns.values)
    for col in df_columns:
        ax = plt.figure(figsize=(14,7))
        ax = sns.violinplot(x='CLASS', y=df[col], data=df)
        plt.show()

tsne(getX(get_mod_data_original()), getY(get_mod_data_original()), [50], 2)