'''
[Random]
score :  0.25121951219512195
acc :  0.25121951219512195
최적의 매개변수 :  XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.7,
              colsample_bynode=1, colsample_bytree=0.6, gamma=0,
              learning_rate=0.1, max_delta_step=0, max_depth=4,
              min_child_weight=1, missing=None, n_estimators=4500, n_jobs=1,
              nthread=None, objective='multi:softprob', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)
예측값 :  [4 1 5 1 6 0 6 7 6 7]
실제값 :  [6 1 5 4 2 8 1 7 1 2]

[Grid]
score :  0.25609756097560976
최적의 매개변수 :  XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
              colsample_bynode=1, colsample_bytree=0.6, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.001, max_delta_step=0, max_depth=6,
              min_child_weight=1, missing=nan, monotone_constraints='()',
              n_estimators=4500, n_jobs=8, num_parallel_tree=1,
              objective='multi:softprob', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
예측값 :  [4 1 3 9 6 0 1 7 6 7]
실제값 :  [6 1 5 4 2 8 1 7 1 2]
'''

# 1. 파이프라인 추가
# 2. pca수정
# 3. grid best

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
warnings.filterwarnings("ignore")

from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam


# 1. 데이터
train = pd.read_csv('./dacon/computer/data/train.csv', header=0)
test = pd.read_csv('./dacon/computer/data/test.csv', header=0)
submit = pd.read_csv('./dacon/computer/data/submission.csv', header=0)

# drop columns
train2 = train.drop(['id','digit','letter'], axis = 1)
test2 = test.drop(['id','letter'], axis = 1)

# print(df_x.shape)      # (2048, 786)
# print(df_y.shape)      # (2048,)

x = train2.iloc[:,:] 
y = train.loc[:,'digit']

x = x.to_numpy()
y = y.to_numpy()
x_pred = test2.to_numpy()



# 데이터 전처리
pca = PCA(n_components=89)
x = pca.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state=104)
kfold = KFold(n_splits=5, shuffle=True)

# 2. 모델
model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
              colsample_bynode=1, colsample_bytree=0.6, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.001, max_delta_step=0, max_depth=6,
              min_child_weight=1, monotone_constraints='()',
              n_estimators=4500, n_jobs=8, num_parallel_tree=1,
              objective='multi:softprob', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)

pipe = Pipeline([('scaler', StandardScaler()), ('mo', model)])

# 3. 훈련
pipe.fit(x_train, y_train, verbose=1, eval_set=[(x_train, y_train),(x_test, y_test)], early_stopping_rounds=10)

# 4. 평가, 예측
y_pred = pipe.predict(x_test)
score = accuracy_score(y_test, y_pred)
print('accuracy_score : ', score)

y_pre = model.predict(x_test[:10])
print('예측값 : ', y_pre)
print('실제값 : ', y_test[:10])

'''
score :  0.25365853658536586
예측값 :  [4 1 5 9 2 4 6 7 6 7]
실제값 :  [6 1 5 4 2 8 1 7 1 2]

2. /255, accuracy_score:
acc :  0.2146341463414634
예측값 :  [4 1 5 5 2 9 6 0 4 5]
실제값 :  [6 1 5 4 2 8 1 7 1 2]

3. pipeline : 
MinMaxScaler , accuracy_score :  0.24634146341463414
예측값 :  [5 9 4 0 2 5 8 3 5 5]
실제값 :  [6 1 5 4 2 8 1 7 1 2]
StandardScaler, accuracy_score :  0.24634146341463414
예측값 :  [3 9 0 0 2 0 8 8 5 5]
실제값 :  [6 1 5 4 2 8 1 7 1 2]
[0.24634146 0.24634146]

4. [random_best]pca cumsum >= 0.98   
accuracy_score :  0.23902439024390243
예측값 :  [4 2 2 3 2 5 8 3 9 5]
실제값 :  [6 1 5 4 2 8 1 7 1 2]

5. [random_best]pca cumsum >= 0.97
accuracy_score :  0.275609756097561
예측값 :  [5 2 0 3 2 0 8 9 5 5]
실제값 :  [6 1 5 4 2 8 1 7 1 2]

6. [grid_best]pca cumsum >= 0.97
accuracy_score :  0.24146341463414633
예측값 :  [5 9 2 8 2 5 8 3 0 8]
실제값 :  [6 1 5 4 2 8 1 7 1 2]

7. [grid_best]pca cumsum >= 0.98
accuracy_score :  0.22926829268292684
예측값 :  [5 2 2 3 2 0 2 3 0 4]
실제값 :  [6 1 5 4 2 8 1 7 1 2]
'''

