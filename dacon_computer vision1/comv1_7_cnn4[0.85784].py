# 	0.85784/ 204등

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
from tensorflow.python.keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

# 1. 데이터
train = pd.read_csv('./dacon/computer/data/train.csv', header=0)
test = pd.read_csv('./dacon/computer/data/test.csv', header=0)
submit = pd.read_csv('./dacon/computer/data/submission.csv', header=0)

# print(train)
# print(train.shape)  # (2048, 787)
# print(test.shape)   # (20480, 786)
# print(train.columns)
# print(test.columns)

# object -> int64 형 변환
train['letter'] = train['letter'].replace({'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,
                                        'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,
                                        'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,
                                        'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,'Z':25})
test['letter'] = test['letter'].replace({'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,
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

# train2 = train.drop(['digit'], axis = 1)      # axis = 0 : 행제거, axis = 1 : 열제거
# test2 = test

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

# pca = PCA(n_components=134)
# pca.fit_transform(x)


# 데이터 전처리
# pca = PCA(n_components=98)
# x = pca.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state=104)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, shuffle = True, random_state=104)


x_train = x_train/255.
x_test = x_test/255.
x_val = x_val/255.
x_pred = x_pred/255.

print(x_train.shape, x_test.shape)  # (1638, 784) (410, 784)
print(y_train.shape, y_test.shape)  # (1638,) (410,)
print(x_pred.shape)                 # (20480, 784)


# x_train = x_train.reshape(-1, 28, 1, 28)        # [[1.82707202 0.46585366]]
# x_test = x_test.reshape(-1, 28, 1, 28)       
# x_val = x_val.reshape(-1, 28, 1, 28)
# x_pred = x_pred.reshape(-1, 28, 1, 28)
# x_train = x_train.reshape(-1, 1, 28, 28)        # [[1.83687949 0.4292683 ]]
# x_test = x_test.reshape(-1, 1, 28, 28)       
# x_val = x_val.reshape(-1, 1, 28, 28) 
# x_pred = x_pred.reshape(-1, 1, 28, 28) 
x_train = x_train.reshape(-1, 28, 28, 1)        # [[1.05655754 0.71951222]]
x_test = x_test.reshape(-1,28, 28, 1)     
x_val = x_val.reshape(-1, 28, 28, 1)  
x_pred = x_pred.reshape(-1, 28, 28, 1)   


# ImageDatagenerator & data augmentation
idg = ImageDataGenerator(height_shift_range=(-1,1),width_shift_range=(-1,1))
# idg = ImageDataGenerator(width_shift_range=5,height_shift_range=5,rotation_range=10,zoom_range=0.05)  
idg2 = ImageDataGenerator()
train_generator = idg.flow(x_train,y_train,batch_size=8)
test_generator = idg2.flow(x_test,y_test)
val_generator = idg2.flow(x_val,y_val)
pred_generator = idg2.flow(x_pred,shuffle=False)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

# 2. 모델구성
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size=3, activation='relu',  input_shape =(x_train.shape[1],x_train.shape[2],x_train.shape[3]), strides=1, padding='same'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(filters = 64, kernel_size=3, activation='relu', strides=1, padding='same'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.4))
model.add(Conv2D(filters = 120, kernel_size=3, activation='relu', strides=1, padding='same'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.4))
model.add(Conv2D(filters = 140, kernel_size=3, activation='relu', strides=1, padding='same'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))
# model.add(Dense(283, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(120, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
opti = [ Nadam(learning_rate=0.001)]       #, 'Adam', 'Adadelta', 'Adamax', 'Adagrad', 'RMSprop', 'SGD', 'Nadam']
loss_list = []

early = EarlyStopping(monitor='val_acc', patience=20, mode= 'auto')
lr = ReduceLROnPlateau(monitor='val_acc', patience=5, factor=0.9, verbose=1) 
modelpath = './dacon/computer/modelcheckpoint/comv1_7_cnn_{epoch:02d}_{val_acc:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_acc', save_best_only=True, mode='auto')

for i in opti:
    model.compile(loss='sparse_categorical_crossentropy', optimizer=i, metrics='acc')
    model.fit_generator(train_generator, epochs=1000,validation_data=val_generator , callbacks=[early, lr, cp])

    # 4. 평가, 예측
    loss = model.evaluate(test_generator)

    loss_list.append(loss)

# model.load_weights('../data/h5/k52_1_weight.h5')
model.save('./dacon/computer/h5/comv1_7_cnn4.h5')

loss_list = np.array(loss_list)
loss_list = loss_list.reshape(-1,2)
print(loss_list)



# submit
y_pred = model.predict_generator(pred_generator)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)
print(y_pred.shape)

submit.loc[:,'digit'] = y_pred

submit.to_csv('./dacon/computer/comv1_7_cnn4.csv',index=False)

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

(-1, 28, 1, 28) : [[1.9383204  0.33414635]]

모델 정리 : [[2.01443338 0.36585367]]
        [[1.70905375 0.38780487]]
        model = Sequential()
        model.add(Conv2D(filters = 32, kernel_size=3, activation='relu',  input_shape =(x_train.shape[1],x_train.shape[2],x_train.shape[3]), strides=2, padding='same'))
        model.add(Dropout(0.4))
        model.add(Conv2D(filters = 32, kernel_size=3, activation='relu', strides=2, padding='same'))
        model.add(Dropout(0.4))
        model.add(Conv2D(filters = 32, kernel_size=3, activation='relu', strides=2, padding='same'))
        model.add(Dropout(0.4))
        model.add(Conv2D(filters = 32, kernel_size=3, activation='relu', strides=2, padding='same'))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.5))
        # model.add(Dense(283, activation='relu'))
        # model.add(Dropout(0.4))
        # model.add(Dense(120, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        [[2.35133243 0.44146341]] -> 드롭제외

        [[1.82707202 0.46585366]] -> lr조절

        [[1.05655754 0.71951222]] -> Nadam(learning_rate=0.0005), factor=0.5
        [[0.84353995 0.74146342]]

        [[0.86561829 0.82682925]] -> Nadam(learning_rate=0.001), factor=0.9, strides=1
 
        [[0.75785494 0.84878051]]    
        model = Sequential()
        model.add(Conv2D(filters = 32, kernel_size=3, activation='relu',  input_shape =(x_train.shape[1],x_train.shape[2],x_train.shape[3]), strides=1, padding='same'))
        # model.add(Dropout(0.2))
        model.add(Conv2D(filters = 64, kernel_size=3, activation='relu', strides=1, padding='same'))
        # model.add(Dropout(0.4))
        model.add(Conv2D(filters = 120, kernel_size=3, activation='relu', strides=2, padding='same'))
        # model.add(Dropout(0.4))
        model.add(Conv2D(filters = 140, kernel_size=3, activation='relu', strides=2, padding='same'))
        # model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.5))
        # model.add(Dense(283, activation='relu'))
        # model.add(Dropout(0.4))
        # model.add(Dense(120, activation='relu'))
        model.add(Dense(10, activation='softmax'))
'''


