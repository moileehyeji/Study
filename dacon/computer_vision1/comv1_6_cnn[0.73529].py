# 0.73529/ 335등

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
from tensorflow.keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, Flatten, Dropout
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

# df_x = train.drop(['digit'], axis = 1)      # axis = 0 : 행제거, axis = 1 : 열제거
# df_y = train.loc[:, 'digit']

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
# pca = PCA(n_components=134)
# x = pca.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state=104)
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, shuffle = True, random_state=66)


x_train = x_train/255.
x_test = x_test/255.
x_pred = x_pred/255.

print(x_train.shape, x_test.shape)  # (1638, 784) (410, 784)
print(y_train.shape, y_test.shape)  # (1638,) (410,)
print(x_pred.shape)                 # (20480, 784)


x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
x_pred = x_pred.reshape(-1, 28, 28, 1)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


model = Sequential()
model.add(Conv2D(filters = 500, kernel_size=2, input_shape =(28, 28, 1), 
                    strides=2, padding='same'))
model.add(Dropout(0.4))
model.add(Conv2D(filters = 300, kernel_size=3, strides=2, padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(filters = 200, kernel_size=3, strides=2, padding='same'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(283, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(120, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
opti = ['Adam', 'Adadelta', 'Nadam']      #, 'Adamax', 'Adagrad', 'RMSprop', 'SGD', 'Nadam']
loss_list = []

early = EarlyStopping(monitor='val_acc', patience=20, mode= 'auto')
lr = ReduceLROnPlateau(monitor='val_acc', patience=5, factor=0.5, verbose=1) 
modelpath = './dacon/computer/modelcheckpoint/comv1_dnn_{epoch:02d}_{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_acc', save_best_only=True, mode='auto')

for i in opti:
    model.compile(loss='categorical_crossentropy', optimizer=i, metrics='acc')
    model.fit(x_train, y_train, epochs=1000, batch_size=200, validation_split=0.3 ,callbacks=[early, lr, cp])

    # 4. 평가, 예측
    loss = model.evaluate(x_test, y_test)

    loss_list.append(loss)

loss_list = np.array(loss_list)
loss_list = loss_list.reshape(-1,2)
print(loss_list)

# submit
y_pred = model.predict(x_pred)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)
print(y_pred.shape)

submit.loc[:,'digit'] = y_pred

submit.to_csv('./dacon/computer/comv1_6_cnn.csv',index=False)

'''
x_pre = x_test[:10]
y_pre = model.predict(x_pre)
y_pre = np.argmax(y_pre, axis=1)
y_test_pre = np.argmax(y_test[:10], axis=1)
print('y_pred[:10] : ', y_pre)
print('y_test[:10] : ', y_test_pre)
'''

'''
[[3.60004783 0.27560976]
 [3.60345221 0.27317074]
 [4.60130692 0.28048781]]
'''


