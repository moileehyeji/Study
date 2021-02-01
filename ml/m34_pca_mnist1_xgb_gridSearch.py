# m31로 만든 0.95이상의 n_conponent = ?를 사용하여 
# gridSearch and RandomSearch + xgb 모델을 만들것

# mnist dnn 보다 성능 좋게 만들어라!!
# cnn과 비교!!

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from xgboost import XGBClassifier

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)

print(x.shape)      #(70000, 28, 28)

# ValueError: Found array with dim 3. Estimator expected <= 2.
x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

x_train , x_test, y_train, y_test  = train_test_split(x,y, test_size = 0.2, shuffle=True, random_state = 66)

pca = PCA(n_components=154)  # 0.95 -> 선택할 차원의 수 : 154,    1.0-> 선택할 차원의 수 : 713
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

kfold = KFold(n_splits=5, shuffle = True)

# parameters       
parameters = [
    {'n_estimators':[100, 200, 300], 'learning_rate':[0.1, 0.3, 0.001, 0.01], 'max_depth': [4,5,6]},
    {'n_estimators':[90, 100, 110], 'learning_rate':[0.1, 0.001, 0.01], 'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1]},
    {'n_estimators':[90,110], 'learning_rate':[0.1, 0.001, 0.5],  'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1], 'colsample_bylevel': [0.6, 0.7, 0.9]}
]

# 2. 모델구성
model = RandomizedSearchCV(XGBClassifier(n_jobs = 8,use_label_encoder=False), parameters, cv=kfold)      #--> 기본 성능은 통상적으로 좋지 않다. 

# 3. 훈련
model.fit(x_train, y_train, eval_metric='mlogloss', verbose = True, eval_set=[(x_train, y_train), (x_test, y_test)])

# 4. 평가, 예측
score = model.score(x_test, y_test)

print('최적의 파라미터 : ', model.best_estimator_)
print('score          : ', score)

'''
mnist_CNN : 
[0.15593186020851135, 0.9835000038146973]
y_pred[:10] :  [7 2 1 0 4 1 4 9 5 9]
y_test[:10] :  [7 2 1 0 4 1 4 9 5 9]

mnist_DNN : 
[0.28995245695114136, 0.9696999788284302]
y_pred[:10] :  [7 2 1 0 4 1 4 9 5 9]
y_test[:10] :  [7 2 1 0 4 1 4 9 5 9]

pca_mnist_DNN (cumsum >= 0.95): 
[0.4257657527923584, 0.9498571157455444]
y_pred[:10] :  [0 4 8 0 5 9 8 2 1 0]
y_test[:10] :  [0 4 8 0 5 9 8 2 1 0]

pca_mnist_DNN (cumsum >= 1.0)
[0.5437219738960266, 0.9520714282989502]
y_pred[:10] :  [0 4 8 0 5 9 8 2 1 0]
y_test[:10] :  [0 4 8 0 5 9 8 2 1 0]

pca_mnist_XGB (cumsum >= 0.95): 
score :  0.9630714285714286

pca_mnist_XGB (cumsum >= 1.0): 
score :  0.9584285714285714

pca_mnist_XGB_SearchCV (cumsum >= 0.95):
최적의 파라미터 :  XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.7,
              colsample_bynode=1, colsample_bytree=0.9, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.5, max_delta_step=0, max_depth=5,
              min_child_weight=1, missing=nan, monotone_constraints='()',
              n_estimators=90, n_jobs=8, num_parallel_tree=1,
              objective='multi:softprob', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', use_label_encoder=False,              validate_parameters=1, verbosity=None)
score          :  0.9616428571428571
'''

