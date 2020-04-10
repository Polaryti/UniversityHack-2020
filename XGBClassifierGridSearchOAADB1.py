from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import XGB_RandomForestBasic_bunyol

#FALTA OBTENER LAS X y para la línea 50, usar XGB_RandomForestBasic_bunyol.get_modelar_data()
estimator = XGBClassifier(
    random_state=420,
    objective= 'binary:logistic',
    tree_method='gpu_hist',
    eval_metric="logloss",
    nthread=4,
    seed=42
)

#Generate lists of floats for each parameter
def gen(start, stop, increment):
  gen_list = []
  while start < stop:
    gen_list.append(start)
    start += increment
  return gen_list

params = [(0.05, 0.4, 0.05), (0.05, 0.6, 0.05), (1, 10, 1), (50, 1000, 20), (0.1, 1.0, 0.05), (0.0, 1.0, 0.05), (0.0, 0.4, 0.05), (0.0, 2.0, 0.1)]
res = []
for i in range(len(params)):
  res.append(gen(params[i][0], params[i][1], params[i][2]))

for i in res:
  print(i)
parameters = {
    'colsample_bytree' : res[0],
    'subsample' : res[1],
    'max_depth': res[2],
    'n_estimators': res[3],
    'learning_rate': res[4],
    'gamma' : res[5],
    'reg_alpha' : res[6],
    'reg_lambda' : res[7]
}

grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring = 'roc_auc',
    n_jobs = 15,
    cv = 10,
    verbose=True
)

grid_search.fit(X, Y)

print(grid_search.best_estimator_)