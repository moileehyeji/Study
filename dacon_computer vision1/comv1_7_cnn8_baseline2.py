# 	0.85784/ 204등

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
warnings.filterwarnings("ignore")

from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, Flatten, Dropout, Activation, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam
from tensorflow.python.keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

def modeling():
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

    model.compile(loss='sparse_categorical_crossentropy', optimizer=opti, metrics='acc')

    return model
'''
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
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state=104)
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, shuffle = True, random_state=104)

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
idg2 = ImageDataGenerator()
train_generator = idg.flow(x_train,y_train,batch_size=10)
test_generator = idg2.flow(x_test,y_test)
val_generator = idg2.flow(x_val,y_val)
pred_generator = idg2.flow(x_pred,shuffle=False)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)
'''
# 1. 데이터
train = pd.read_csv('./dacon/computer/data/train.csv', header=0)    # (2048, 787)
test = pd.read_csv('./dacon/computer/data/test.csv', header=0)      # (20480, 786)
submit = pd.read_csv('./dacon/computer/data/submission.csv', header=0)


# x (-1, 28, 28, 1), y (2048, 10), test (-1, 28, 28, 1)
x_train = train.drop(['id','digit','letter'], axis = 1).values 
x_train = x_train.reshape(-1, 28, 28, 1)
x_train = np.where((x_train<=20)&(x_train!=0) ,0.,x_train)      #0 < x_train <=20 0 으로 바꿔
x_train = x_train/255.
x_train = x_train.astype('float32')

y_train = train['digit']  
# y_train = to_categorical(y)                                 
# y_train = np.zeros((len(y), len(y.unique())))  # 총 행의수 , 10(0~9)
# for i, digit in enumerate(y):   # y전처리 for문
#     y_train[i, digit] = 1       # (2048, 10)

x_test = test.drop(['id', 'letter'], axis=1).values
x_test = x_test.reshape(-1, 28, 28, 1)
x_test = np.where((x_test<=20)&(x_test!=0) ,0.,x_test)
x_test = x_test/255
x_test = x_test.astype('float32')

# ImageDataGenerator
# idg = ImageDataGenerator(height_shift_range=(-1,1),width_shift_range=(-1,1))
# idg2 = ImageDataGenerator()
datagen = ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        # zca_whitening=True,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.15,
        rotation_range = 10,
        validation_split=0.2
        )
valgen = ImageDataGenerator()

# cross validation
n_splits = 40
skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)

val_acc_max = []
result = 0
nth = 0
Fold = 1

for train_index, valid_index in skf.split(x_train, y_train) :

    opti = Nadam(learning_rate=0.001)
    epochs = 1000

    early = EarlyStopping(monitor='val_acc', patience=20, mode= 'auto')
    lr = ReduceLROnPlateau(monitor='val_acc', patience=5, factor=0.9, verbose=1) 
    modelpath = './dacon/computer/modelcheckpoint/comv1_7_cnn8_base_2{epoch:02d}_{val_acc:.4f}.hdf5'
    cp = ModelCheckpoint(filepath=modelpath, monitor='val_acc', save_best_only=True, mode='auto')

    X_train = x_train[train_index]
    X_valid = x_train[valid_index]    
    Y_train = y_train[train_index]
    Y_valid = y_train[valid_index]

    train_generator = datagen.flow(X_train,Y_train,batch_size=8)
    valid_generator = valgen.flow(X_valid,Y_valid)
    test_generator = valgen.flow(x_test,shuffle=False)

    model = modeling()

    # learning_history  = model.fit_generator(train_generator,epochs=epochs, validation_data=valid_generator,callbacks=[early, lr, cp])
    learning_history  = model.fit(train_generator, epochs=epochs, validation_data=valid_generator, callbacks=[early, lr, cp])

    model.save_weights('./dacon/computer/h5/baseline_weight.h5')
    model.save('./dacon/computer/h5/baseline_model.h5')
    
    result += model.predict_generator(test_generator,verbose=True)/40
    
    # save val_loss
    hist = pd.DataFrame(learning_history.history)
    val_acc_max.append(hist['val_acc'].max())
    
    nth += 1
    print(nth, '번째 학습을 완료했습니다.')


# submit
submit['digit'] = np.argmax(result, axis=1)
submit.to_csv('./dacon/computer/comv7_8_base2_.csv',index=False)

print('val_acc_max      : ', val_acc_max)
print('val_acc_max 평균 :', np.mean(val_acc_max))

'''
val_acc_max      :  [0.8269230723381042, 0.9038461446762085, 0.9038461446762085, 0.9230769276618958, 0.9807692170143127, 0.8653846383094788, 0.942307710647583, 
                    0.8653846383094788, 0.9019607901573181, 0.8823529481887817, 0.9607843160629272, 0.9411764740943909, 0.9411764740943909, 0.9411764740943909, 
                    0.9019607901573181, 0.8823529481887817, 0.9607843160629272, 0.8823529481887817, 0.9019607901573181, 0.9019607901573181, 0.9019607901573181, 
                    0.9411764740943909, 0.8823529481887817, 0.8823529481887817, 0.9019607901573181, 0.8627451062202454, 0.8627451062202454, 0.9607843160629272, 
                    0.9215686321258545, 0.9019607901573181, 0.9019607901573181, 0.9411764740943909, 0.8627451062202454, 0.9607843160629272, 0.9607843160629272, 
                    0.8627451062202454, 0.8235294222831726, 0.9607843160629272, 0.9607843160629272, 0.8823529481887817]
val_acc_max 평균 : 0.908719839155674
'''


''' 

# 2. 모델
model = modeling()

# 3. 컴파일, 훈련
opti = [ Nadam(learning_rate=0.001)]    #, Adam(learning_rate=0.001), Adadelta(learning_rate=0.001),
                                        # Adamax(learning_rate=0.001), Adagrad(learning_rate=0.001), RMSprop(learning_rate=0.001), SGD(learning_rate=0.001)]
loss_list = []

early = EarlyStopping(monitor='val_acc', patience=20, mode= 'auto')
lr = ReduceLROnPlateau(monitor='val_acc', patience=5, factor=0.7, verbose=1) 
modelpath = './dacon/computer/modelcheckpoint/comv1_7_cnn_{epoch:02d}_{val_acc:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_acc', save_best_only=True, mode='auto')

for i in opti:
    model.compile(loss='sparse_categorical_crossentropy', optimizer=i, metrics='acc')
    model.fit_generator(train_generator, epochs=1000,validation_data=val_generator , callbacks=[early, lr, cp])

    # 4. 평가, 예측
    loss = model.evaluate(test_generator)

    loss_list.append(loss)

# model.load_weights('../data/h5/k52_1_weight.h5')
model.save('./dacon/computer/h5/comv1_7_cnn8.h5')

loss_list = np.array(loss_list)
loss_list = loss_list.reshape(-1,2)
print(loss_list)

# submit
y_pred = model.predict_generator(pred_generator)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)
print(y_pred.shape)

submit.loc[:,'digit'] = y_pred

submit.to_csv('./dacon/computer/comv1_7_cnn8.csv',index=False)

 '''

'''
ImageDatagenerator 사용 : 
[[0.34506825 0.91219515]Nadam
 [0.41096631 0.89268291]Adam
 [0.40027758 0.88780487]Adadelta
 [0.36024177 0.90975612]Adamax
 [0.36661682 0.90975612]Adagrad
 [0.46746987 0.88780487]RMSprop
 [0.44311273 0.90975612]]SGD

 [[0.2616542  0.91463417]]

 사용 xx : 
 [[0.46529353 0.88292682]]
'''