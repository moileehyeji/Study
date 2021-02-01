# m31로 만든 1.0이상의 n_conponent = ?를 사용하여 
# xgb 모델을 만들것

# mnist dnn 보다 성능 좋게 만들어라!!
# cnn과 비교!!

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)

print(x.shape)      #(70000, 28, 28)

# ValueError: Found array with dim 3. Estimator expected <= 2.
x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

pca = PCA(n_components=713)  # 0.95 -> 선택할 차원의 수 : 154,    1.0-> 선택할 차원의 수 : 713
x = pca.fit_transform(x)

x_train , x_test, y_train, y_test  = train_test_split(x,y, test_size = 0.2, shuffle=True, random_state = 66)

# 2. 모델구성
model = XGBClassifier(n_jobs = 8,use_label_encoder=False) 

# 3. 훈련
model.fit(x_train, y_train, eval_metric='mlogloss',verbose = True,eval_set=[(x_train, y_train), (x_test, y_test)])


# 4. 평가, 예측
score = model.score(x_test, y_test)

print('score : ', score)

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
'''

