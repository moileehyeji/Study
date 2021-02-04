import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import tensorflow as tf
warnings.filterwarnings("ignore")

from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, RepeatedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, Flatten, Dropout, Activation, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam
from tensorflow.python.keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers

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

    opti =  Nadam(learning_rate=0.001)   #, Adam(learning_rate=0.001), Adadelta(learning_rate=0.001),
                                        # Adamax(learning_rate=0.001), Adagrad(learning_rate=0.001), RMSprop(learning_rate=0.001), SGD(learning_rate=0.001)]

    model.compile(loss="categorical_crossentropy",
                optimizer=RMSprop(lr=initial_learningrate),
                metrics=['acc'])

    return model

def create_model() :
    
  effnet = tf.keras.applications.EfficientNetB3(
      include_top=True,
      weights=None,
      input_shape=(28,28,1),
      classes=10,
      classifier_activation="softmax",
  )
  model = Sequential()
  model.add(effnet)


  model.compile(loss="categorical_crossentropy",
              optimizer=RMSprop(lr=initial_learningrate),
              metrics=['acc'])
  return model




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

y = train['digit']                                     
y_train = np.zeros((len(y), len(y.unique())))  # 총 행의수 , 10(0~9)
for i, digit in enumerate(y):   # y전처리 for문
    y_train[i, digit] = 1       # (2048, 10)

x_test = test.drop(['id', 'letter'], axis=1).values
x_test = x_test.reshape(-1, 28, 28, 1)
x_test = np.where((x_test<=20)&(x_test!=0) ,0.,x_test)
x_test = x_test/255
x_test = x_test.astype('float32')


datagen = ImageDataGenerator(
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.15,
        rotation_range = 10,
        validation_split=0.2)
valgen = ImageDataGenerator()



# 0.8 / 0.2 로 train/vali 데이터로 학습시켜 앙상블
kfold = RepeatedKFold(n_splits=5, n_repeats=10, random_state=40)
cvscores = []
Fold = 1
results = np.zeros((20480,10) )

# # Modelcheckpoint, EarlyStopping을 이용해서 epoch 값을 조절하면서 val_accuracy 기준으로 모델을 저장하였습니다.
# initial_learningrate=2e-3  
# def lr_decay(epoch):#lrv
#     return initial_learningrate * 0.99 ** epoch


results = np.zeros( (20480,10),dtype=np.float32)


for train, val in kfold.split(x_train) : 

    initial_learningrate=2e-3  
    early = EarlyStopping(monitor='acc', patience=20, mode= 'auto')
    lr = ReduceLROnPlateau(monitor='acc', patience=5, factor=0.9, verbose=1) 
    modelpath = './dacon/computer/modelcheckpoint/comv3_{epoch:02d}_{acc:.4f}.hdf5'
    cp = ModelCheckpoint(filepath=modelpath, monitor='acc', save_best_only=True, mode='auto')

    print ('Fold: ',Fold)

    X_train = x_train[train]
    X_val = x_train[val]
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    Y_train = y_train[train]
    Y_val = y_train[val]

    model = modeling()
    # model =  create_model()

    training_generator = datagen.flow(X_train, Y_train, batch_size=32,seed=7,shuffle=True)
    validation_generator = valgen.flow(X_val, Y_val, batch_size=32,seed=7,shuffle=True)

    model.fit(training_generator,epochs=1000,callbacks=[early, lr, cp],
               shuffle=True,
               validation_data=validation_generator,
               steps_per_epoch =len(X_train)//32
               )

    del X_train
    del X_val
    del Y_train
    del Y_val

    results += model.predict(x_test)

    Fold += 1

# submit
submit['digit'] = np.argmax(results, axis=1)
submit.to_csv('./dacon/computer/comv3_.csv',index=False)

'''
width_shift_range=0.05, height_shift_range=0.05, -> 
'''