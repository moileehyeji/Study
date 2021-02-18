# 	0.90686/ 118등

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state=104)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, shuffle = True, random_state=104)

x_train = x_train/255.
x_test = x_test/255.
x_val = x_val/255.
x_pred = x_pred/255.

print(x_train.shape, x_test.shape)  # (1638, 784) (410, 784)
print(y_train.shape, y_test.shape)  # (1638,) (410,)
print(x_pred.shape)                 # (20480, 784)

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

model = Sequential()
    
model.add(Conv2D(16,(3,3),activation='relu',input_shape=(28,28,1),padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
    
model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32,(5,5),activation='relu',padding='same')) 
model.add(BatchNormalization())
model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((3,3)))
model.add(Dropout(0.3))
    
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64,(5,5),activation='relu',padding='same')) 
model.add(BatchNormalization())
model.add(MaxPooling2D((3,3)))
model.add(Dropout(0.3))
    
model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(10,activation='softmax'))

# 3. 컴파일, 훈련
opti = [ Nadam(learning_rate=0.001)]       #, 'Adam', 'Adadelta', 'Adamax', 'Adagrad', 'RMSprop', 'SGD', 'Nadam']
loss_list = []

early = EarlyStopping(monitor='val_acc', patience=20, mode= 'auto')
lr = ReduceLROnPlateau(monitor='val_acc', patience=5, factor=0.6, verbose=1) 
modelpath = './dacon/computer/modelcheckpoint/comv1_7_cnn_{epoch:02d}_{val_acc:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_acc', save_best_only=True, mode='auto')

for i in opti:
    model.compile(loss='sparse_categorical_crossentropy', optimizer=i, metrics='acc')
    model.fit_generator(train_generator, epochs=1000,validation_data=val_generator , callbacks=[early, lr, cp])

    # 4. 평가, 예측
    loss = model.evaluate(test_generator, batch_size=8)

    loss_list.append(loss)

# model.load_weights('../data/h5/k52_1_weight.h5')
model.save('./dacon/computer/h5/comv1_7_cnn5.h5')

loss_list = np.array(loss_list)
loss_list = loss_list.reshape(-1,2)
print(loss_list)



# submit
y_pred = model.predict_generator(pred_generator)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)
print(y_pred.shape)

submit.loc[:,'digit'] = y_pred

submit.to_csv('./dacon/computer/comv1_7_cnn5.csv',index=False)


""" 
[[0.75785494 0.84878051]]  

[[0.33662117 0.89512193]]

"""

