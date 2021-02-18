'''
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
'''

# 1. 파이프라인 추가

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

print(train)
# print(train.shape)  # (2048, 787)
# print(test.shape)   # (20480, 786)
# print(train.columns)
# print(test.columns)

# object -> int64 형 변환
train['letter'] = train['letter'].replace({'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,
                                        'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,
                                        'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,
                                        'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,'Z':25})
# train['letter'] = pd.to_numeric(train['letter'])
# train["letter"].astype(np.int)
# print(train.info())


'''
train.columns:  Index(['id', 'digit', 'letter', '0', '1', '2', '3', '4', '5', '6',
                        ...
                        '774', '775', '776', '777', '778', '779', '780', '781', '782', '783'],
                        dtype='object', length=787)
test.columns :  Index(['id', 'letter', '0', '1', '2', '3', '4', '5', '6', '7',
                        ...
                        '774', '775', '776', '777', '778', '779', '780', '781', '782', '783'],
                        dtype='object', length=786)
'''

df_x = train.drop(['digit'], axis = 1)
df_y = train.loc[:, 'digit']

# print(df_x.shape)      # (2048, 786)
# print(df_y.shape)      # (2048,)

x = df_x.to_numpy()
y = df_y.to_numpy()



# 데이터 전처리
'''
# PCA
pca = PCA()
pca.fit(x)

cumsum = np.cumsum(pca.explained_variance_ratio_)
print('cumsum : ', cumsum)     

d = np.argmax(cumsum >= 0.95) + 1
print('cumsum >= 0.95   :', cumsum >= 0.95)
print('선택할 차원의 수  : ', d)    # 선택할 차원의 수  :  89

import matplotlib.pyplot as plt
plt.plot(cumsum)        
plt.grid()
# plt.show()
'''
pca = PCA(n_components=89)
x = pca.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state=104)
kfold = KFold(n_splits=5, shuffle=True)

# 2. 모델
scalers = [MinMaxScaler(), StandardScaler()]
scores = []
for i in scalers :
    model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.7,
                colsample_bynode=1, colsample_bytree=0.6, gamma=0,
                learning_rate=0.1, max_delta_step=0, max_depth=4,
                min_child_weight=1, missing=None, n_estimators=4500, n_jobs=1,
                nthread=None, objective='multi:softprob', random_state=0,
                reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                silent=None, subsample=1, verbosity=1)

    pipe = Pipeline([('scaler', i), ('mo', model)])

    # 3. 훈련
    pipe.fit(x_train, y_train)

    # 4. 평가, 예측
    y_pred = pipe.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    scores.append(score)
    print('accuracy_score : ', score)

    y_pre = model.predict(x_test[:10])
    print('예측값 : ', y_pre)
    print('실제값 : ', y_test[:10])

scores = np.array(scores)
scores = scores.reshape(-1,)
print(scores)

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
'''

