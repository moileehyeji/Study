from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np

dataset = load_boston()
x = dataset.data
y = dataset['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 66)

##########################################################################
model = XGBRegressor(n_estimators = 100000, learning_rate = 0.01,
                        tree_method = 'gpu_hist',           # gpu 훈련
                        predictor = 'gpu_predictor',        # gpu 예측, cpu_predictor: cpu 예측
                        gpu_id = 0
                        )
##########################################################################

model.fit(x_train, y_train, verbose=1, eval_metric=['rmse'], 
                    eval_set=[(x_train, y_train), (x_test, y_test)], 
                    early_stopping_rounds=10000)

aaa = model.score(x_test, y_test)
print('model.score : ', aaa)