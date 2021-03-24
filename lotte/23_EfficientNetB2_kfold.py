import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4, EfficientNetB2
from tensorflow.keras.applications.efficientnet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, GlobalMaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Dense, Activation, Dropout
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import VGG19, MobileNet, ResNet101
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection._split import StratifiedKFold


#데이터 지정 및 전처리
x = np.load("C:/data/lotte/npy/128_project_x.npy",allow_pickle=True)
x_pred = np.load('C:/data/lotte/npy/128_test.npy',allow_pickle=True)
y = np.load("C:/data/lotte/npy/128_project_y.npy",allow_pickle=True)

# print(x.shape, x_pred.shape, y.shape)   #(48000, 128, 128, 3) (72000, 128, 128, 3) (48000, 1000)

x = preprocess_input(x) # (48000, 255, 255, 3)
x_pred = preprocess_input(x_pred)   # 



idg = ImageDataGenerator(
    # rotation_range=10, acc 하락
    width_shift_range=(-1,1),  
    height_shift_range=(-1,1), 
    rotation_range=40, 
    # shear_range=0.2)    # 현상유지
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

idg2 = ImageDataGenerator()

""" x_train, x_valid, y_train, y_valid = train_test_split(x,y, train_size = 0.9, shuffle = True, random_state=66)

train_generator = idg.flow(x_train,y_train,batch_size=64, seed = 2048)
# seed => random_state
valid_generator = idg2.flow(x_valid,y_valid)
# test_generator = idg2.flow(x_pred) """

#==============================================kfold
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=66)

i = 0
val_loss_min = []
val_acc_max = []

y = np.argmax(y, axis=1)

for train, valid in skf.split(x,y):
    i += 1
    x_train = x[train]
    x_valid = x[valid]
    y_train = y[train]
    y_valid = y[valid]

    #x_train, x_valid, y_train, y_valid = train_test_split(x,y, train_size = 0.8, shuffle = True, random_state=42)

    train_generator = idg.flow(x_train,y_train,batch_size=32, seed = 2048)
    # seed => random_state
    valid_generator = idg2.flow(x_valid,y_valid)
    # test_generator = idg2.flow(x_pred)

    mc = ModelCheckpoint(f'C:/data/lotte/h5/23_EfficientNetB2_kfold{i}.h5',save_best_only=True, verbose=1)
    # efficientnet = EfficientNetB4(include_top=False,weights='imagenet',input_shape=x_train.shape[1:])
    mobile = EfficientNetB2(include_top=False,weights='imagenet',input_shape=x_train.shape[1:])
    mobile.trainable = True
    a = mobile.output
    a = GlobalAveragePooling2D() (a)
    # a = Flatten() (a)
    a = Dense(2048, activation= 'swish') (a)
    a = Dropout(0.2) (a)
    a = Dense(2048, activation= 'swish') (a)
    a = Dropout(0.2) (a)
    a = Dense(1024, activation= 'swish') (a)
    a = Dropout(0.2) (a)
    a = Dense(1024, activation= 'swish') (a)
    a = Dropout(0.2) (a)
    a = Dense(1000, activation= 'softmax') (a)

    model = Model(inputs = mobile.input, outputs = a)

    early_stopping = EarlyStopping(patience= 20)
    lr = ReduceLROnPlateau(patience= 10, factor=0.5)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['acc'])
    # learning_history = model.fit_generator(train_generator,epochs=200, steps_per_epoch= len(x_train) / 32,
    #     validation_data=valid_generator, callbacks=[early_stopping,lr,mc])

    # predict
    model.load_weights(f'C:/data/lotte/h5/23_EfficientNetB2_kfold{i}.h5')
    result = model.predict(x_pred,verbose=True)


    # 제출생성
    sub = pd.read_csv('C:/data/lotte/csv/sample.csv')
    sub['prediction'] = np.argmax(result,axis = 1)
    # sub.to_csv('C:/Study/lotte/data/15_relu.csv',index=False)
    sub.to_csv(f'C:/data/lotte/csv/23_{i}.csv',index=False)



# =====================================15_relu: 20등 / 74.217
# Epoch 9/200
# 675/675 [==============================] - 166s 246ms/step - loss: 0.0266 - acc: 0.9920 - val_loss: 0.0270 - val_acc: 0.9917

# Epoch 00009: val_loss did not improve from 0.01850
# Epoch 10/200
# 132/675 [====>.........................] - ETA: 2:10 - loss: 0.0194 - acc: 0.9951
#==========================================

# =====================================15_swish: 18등 / 75.249
# Epoch 00018: val_loss did not improve from 0.01504
# Epoch 19/200
# 675/675 [==============================] - 160s 237ms/step - loss: 0.0113 - acc: 0.9967 - val_loss: 0.0194 - val_acc: 0.9933

# Epoch 00019: val_loss did not improve from 0.01504
# Epoch 20/200
# 126/675 [====>.........................] - ETA: 2:05 - loss: 0.0137 - acc: 0.9968
#==========================================

#==========================================레이어 늘리기 : 레이어를 늘리면 특성을 저 잘 잡음 16: 19등/ 77.115
# Epoch 00084: val_loss did not improve from 0.00304
# Epoch 85/200
# 675/675 [==============================] - 160s 237ms/step - loss: 7.4497e-04 - acc: 0.9999 - val_loss: 0.0046 - val_acc: 0.9987

# Epoch 00085: val_loss did not improve from 0.00304
# 2250/2250 [==============================] - 70s 30ms/step
#==========================================

#==========================================tta
# 18_ttaOOOOO0 : 23등 / 77.389
# 18_ttaOOOOO1 : 23등 / 77.389
# tensorflow.python.framework.errors_impl.InternalError: Dst tensor is not initialized.
#==========================================

#==========================================23_1 kfold : 76.737점