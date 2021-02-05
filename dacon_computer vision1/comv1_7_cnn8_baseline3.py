# 	0.85784/ 204등

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
warnings.filterwarnings("ignore")

from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, RepeatedKFold, StratifiedKFold, cross_val_score, train_test_split
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

    model.compile(loss='categorical_crossentropy', optimizer=opti, metrics='acc')

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
# y_train = to_categorical(y)                                 
y_train = np.zeros((len(y), len(y.unique())))  # 총 행의수 , 10(0~9)
for i, digit in enumerate(y):   # y전처리 for문
    y_train[i, digit] = 1       # (2048, 10)

x_test = test.drop(['id', 'letter'], axis=1).values
x_test = x_test.reshape(-1, 28, 28, 1)
x_test = np.where((x_test<=20)&(x_test!=0) ,0.,x_test)
x_test = x_test/255
x_test = x_test.astype('float32')

# ImageDataGenerator
# idg = ImageDataGenerator(height_shift_range=(-1,1),width_shift_range=(-1,1))
# idg2 = ImageDataGenerator()
datagen = ImageDataGenerator(
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
# skf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=40)

val_acc_max = []
result = 0
nth = 0
Fold = 1

for train_index, valid_index in skf.split(x_train) :

    opti = Nadam(learning_rate=0.002)
    epochs = 1000

    early = EarlyStopping(monitor='val_acc', patience=20, mode= 'auto')
    lr = ReduceLROnPlateau(monitor='val_acc', patience=5, factor=0.5, verbose=1) 
    modelpath = './dacon/computer/modelcheckpoint/comv1_7_cnn8_base_3{epoch:02d}_{val_acc:.4f}.hdf5'
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

    model.save_weights('./dacon/computer/h5/baseline3_weight.h5')
    model.save('./dacon/computer/h5/baseline3_model.h5')
    
    result += model.predict(test_generator,verbose=True)/50
    
    # save val_loss
    hist = pd.DataFrame(learning_history.history)
    val_acc_max.append(hist['val_acc'].max())
    
    nth += 1
    print(nth, '번째 학습을 완료했습니다.')


# submit
submit['digit'] = np.argmax(result, axis=1)
submit.to_csv('./dacon/computer/comv7_8_base3_.csv',index=False)

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