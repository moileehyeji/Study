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

df_x = train.drop(['digit'], axis = 1)      # axis = 0 : 행제거, axis = 1 : 열제거
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
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, shuffle = True, random_state=66)

x_train = x_train/255
x_test = x_test/255
# x_val = x_val/255

# kfold = KFold(n_splits=5, shuffle=True)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


model = Sequential()
model.add(Dense(10, input_shape=(89,), activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
opti = ['Adam', 'Adadelta', 'Nadam']      #, 'Adamax', 'Adagrad', 'RMSprop', 'SGD', 'Nadam']
loss_list = []

early = EarlyStopping(monitor='loss', patience=20, mode= 'auto')
lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1) 
modelpath = './dacon/computer/modelcheckpoint/comv1_dnn_{epoch:02d}_{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')

for i in opti:
    model.compile(loss='categorical_crossentropy', optimizer=i, metrics='acc')
    model.fit(x_train, y_train, epochs=1000, batch_size=200, validation_split=0.3 ,callbacks=[early, lr, cp])

    # 4. 평가, 예측
    loss = model.evaluate(x_test, y_test)

    loss_list.append(loss)

loss_list = np.array(loss_list)
loss_list = loss_list.reshape(-1,2)
print(loss_list)

'''
[[3.5169909  0.12195122]
 [3.51778173 0.12195122]
 [3.8447206  0.13170731]]
'''

'''
x_pre = x_test[:10]
y_pre = model.predict(x_pre)
y_pre = np.argmax(y_pre, axis=1)
y_test_pre = np.argmax(y_test[:10], axis=1)
print('y_pred[:10] : ', y_pre)
print('y_test[:10] : ', y_test_pre)
'''
'''
[3.32369065284729, 0.13658536970615387]
y_pred[:10] :  [9 9 9 9 0 8 7 7 8 6]
y_test[:10] :  [6 1 5 4 2 8 1 7 1 2]
'''