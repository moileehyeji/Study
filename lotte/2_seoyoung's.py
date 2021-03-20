import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, KFold
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input


#데이터 지정 및 전처리
x = np.load("C:/Study/lotte/data/npy/1_128_x.npy",allow_pickle=True)
x_pred = np.load('C:/Study/lotte/data/npy/1_128_test.npy',allow_pickle=True)
y = np.load("C:/Study/lotte/data/npy/1_128_y.npy",allow_pickle=True)

x = preprocess_input(x)
x_pred = preprocess_input(x_pred)

idg = ImageDataGenerator(
    width_shift_range=(-1,1),   
    height_shift_range=(-1,1),  
    shear_range=0.2) 


idg2 = ImageDataGenerator()

# y = np.argmax(y, axis=1)

from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x,y, train_size = 0.8, shuffle = True, random_state=66)

mc = ModelCheckpoint('C:/Study/lotte/data/h5/2_seoyoungs.h5',save_best_only=True, verbose=1)

train_generator = idg.flow(x_train,y_train,batch_size=24)
# seed => random_state
valid_generator = idg2.flow(x_valid,y_valid)
test_generator = x_pred

from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation
from tensorflow.keras.applications import VGG19, MobileNet


mobile_net = MobileNet(weights="imagenet", include_top=False, input_shape=x_train.shape[1:])

top_model = mobile_net.output
top_model = Flatten()(top_model)
top_model = Dense(512, activation="relu")(top_model)
# top_model = Dropout(0.2)(top_model)
top_model = Dense(1000, activation="softmax")(top_model)

model = Model(inputs=mobile_net.input, outputs = top_model)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
early_stopping = EarlyStopping(patience= 20)
lr = ReduceLROnPlateau(patience= 10, factor=0.5)

model.compile(optimizer=tf.keras.optimizers.Adam(), loss = 'categorical_crossentropy', metrics=['accuracy'])

# learning_history = model.fit_generator(train_generator,epochs=70, 
#     validation_data=valid_generator, callbacks=[early_stopping,lr,mc])


# 학습 완료된 모델 저장
hdf5_file = 'C:/Study/lotte/data/h5/2_seoyoungs.h5'
if os.path.exists(hdf5_file):
    # 기존에 학습된 모델 불러들이기
    model.load_weights(hdf5_file)
else:
    learning_history = model.fit_generator(train_generator,epochs=70, 
    validation_data=valid_generator, callbacks=[early_stopping,lr,mc])
    # 학습한 모델이 없으면 파일로 저장
    model.save_weights(hdf5_file)

# model.load_weights('C:/Study/lotte/data/h5/2_seoyoungs_2.h5')

# predict
result = model.predict(test_generator,verbose=True)
    
print(result.shape)
sub = pd.read_csv('C:/Study/lotte/data/sample.csv')
sub['prediction'] = np.argmax(result,axis = 1)
sub.to_csv('C:/Study/lotte/data/2_2.csv',index=False)