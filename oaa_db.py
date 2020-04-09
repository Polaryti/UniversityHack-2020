import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import recall_score,accuracy_score,confusion_matrix, f1_score, precision_score

import XGB_RandomForestBasic_bunyol

estimar_df = XGB_RandomForestBasic_bunyol.get_estimar_data()
modelar_df = XGB_RandomForestBasic_bunyol.get_modelar_data()

