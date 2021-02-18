import numpy as np
import pandas as pd
import os
import shutil
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
# from cv2 import cv2
import cv2
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from collections import Counter

# 1. 데이터
csv_train  = pd.read_csv('./dacon/computer/data/train.csv')
csv_test = pd.read_csv('./dacon/computer/data/test.csv')
submission = pd.read_csv('./dacon/computer/data/submission.csv')

# train 이미지들과 test 이미지들을 저장해놓을 폴더를 생성합니다.
path_img = './dacon/computer/data/img'
# os.mkdir(f'{path_img}images_train')
# os.mkdir(f'{path_img}images_train/0')
# os.mkdir(f'{path_img}images_train/1')
# os.mkdir(f'{path_img}images_train/2')
# os.mkdir(f'{path_img}images_train/3')
# os.mkdir(f'{path_img}images_train/4')
# os.mkdir(f'{path_img}images_train/5')
# os.mkdir(f'{path_img}images_train/6')
# os.mkdir(f'{path_img}images_train/7')
# os.mkdir(f'{path_img}images_train/8')
# os.mkdir(f'{path_img}images_train/9')
# os.mkdir(f'{path_img}images_test')

# for idx in range(len(csv_train)) :
#     img = csv_train.loc[idx, '0':].values.reshape(28, 28).astype(int)
#     digit = csv_train.loc[idx, 'digit']
#     cv2.imwrite(f'{path_img}/images_train/{digit}/{csv_train["id"][idx]}.png', img)


# for idx in range(len(csv_test)) :
#     img = csv_test.loc[idx, '0':].values.reshape(28, 28).astype(int)
#     cv2.imwrite(f'{path_img}/images_test/{csv_test["id"][idx]}.png', img)



# 모델구성

# 폴더에 저장되어있는 이미지들을 사용하여 학습할 모델을 생성합니다.
# 모델은 3가지이며, 최종 예측값은 최빈값(most frequent value)으로 결정합니다.

# InceptionResNetV2 : 사전훈련된 convolution 신경망
# GlobalAveragePooling2D : 뒤로 갈 수록 추상화되고 함축되는 정보가 feature에 담기게 되는데, 
#                           결과적으로 마지막  feature를 분류기로 사용(https://jetsonaicar.tistory.com/16)     
def modeling():
    model = Sequential()
    model.add(Conv2D(16,(3,3),activation='relu',input_shape=(224, 224, 1),padding='same'))
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

    return model

model = modeling(

)
# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2,
                             rotation_range=10,
                             width_shift_range=0.1,
                             height_shift_range=0.1)

train_generator = datagen.flow_from_directory(f'{path_img}/images_train', target_size=(224,224), color_mode='grayscale', class_mode='categorical', subset='training')
val_generator = datagen.flow_from_directory(f'{path_img}/images_train', target_size=(224,224), color_mode='grayscale', class_mode='categorical', subset='validation')


path_h5 = './dacon/computer/h5'
checkpoint_1 = tf.keras.callbacks.ModelCheckpoint(f'{path_h5}/baseline1,6_model_1.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
checkpoint_2 = tf.keras.callbacks.ModelCheckpoint(f'{path_h5}/baseline1,6_model_2.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
checkpoint_3 = tf.keras.callbacks.ModelCheckpoint(f'{path_h5}/baseline1,6_model_3.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

lr = ReduceLROnPlateau(monitor='val_accuracy', patience=10, factor=0.2, verbose=1) 

model.fit_generator(train_generator, epochs=200, validation_data=val_generator, callbacks=[checkpoint_1, lr])


# 4. 평가, 예측

# 학습한 모델의 accuracy와 val_accuracy를 시각화
plt.plot(model.history.history["accuracy"], label='m1_acc')
plt.plot(model.history.history["val_accuracy"], label='m1_vacc')

plt.legend()
plt.show()


# 결과(예측값) 확인
# model_1 = tf.keras.models.load_model(f'{path_h5}/baseline6_model_1.h5', compile=False)
# model_2 = tf.keras.models.load_model(f'{path_h5}/baseline6_model_1.h5', compile=False)
# model_3 = tf.keras.models.load_model(f'{path_h5}/baseline6_model_1.h5', compile=False)


atagen = ImageDataGenerator(rescale=1./255)
test_generator = datagen.flow_from_directory(f'{path_img}/images_test', 
                        target_size=(224,224), 
                        color_mode='grayscale', 
                        class_mode='categorical', 
                        shuffle=False)


predict_1 = model.predict_generator(test_generator).argmax(axis=1)


# 제출용 csv 파일 생성하기
# 3가지 예측값 중에서 최빈값(most frequent value)을 최종 예측값으로 제출합니다.
submission["digit"] = predict_1


path_submit = './dacon/computer'
submission.to_csv(f'{path_submit}/baseline1,6.csv', index=False)





'''                  
model_1 = tf.keras.applications.InceptionResNetV2(weights=None, include_top=True, input_shape=(224, 224, 1), classes=10)

model_2 = tf.keras.Sequential([
                               tf.keras.applications.InceptionV3(weights=None, include_top=False, input_shape=(224, 224, 1)),
                               tf.keras.layers.GlobalAveragePooling2D(),
                               tf.keras.layers.Dense(1024, kernel_initializer='he_normal'),
                               tf.keras.layers.BatchNormalization(),
                               tf.keras.layers.Activation('relu'),
                               tf.keras.layers.Dense(512, kernel_initializer='he_normal'),
                               tf.keras.layers.BatchNormalization(),
                               tf.keras.layers.Activation('relu'),
                               tf.keras.layers.Dense(256, kernel_initializer='he_normal'),
                               tf.keras.layers.BatchNormalization(),
                               tf.keras.layers.Activation('relu'),
                               tf.keras.layers.Dense(10, kernel_initializer='he_normal', activation='softmax', name='predictions')
                               ])

model_3 = tf.keras.Sequential([
                               tf.keras.applications.Xception(weights=None, include_top=False, input_shape=(224, 224, 1)),
                               tf.keras.layers.GlobalAveragePooling2D(),
                               tf.keras.layers.Dense(1024, kernel_initializer='he_normal'),
                               tf.keras.layers.BatchNormalization(),
                               tf.keras.layers.Activation('relu'),
                               tf.keras.layers.Dense(512, kernel_initializer='he_normal'),
                               tf.keras.layers.BatchNormalization(),
                               tf.keras.layers.Activation('relu'),
                               tf.keras.layers.Dense(256, kernel_initializer='he_normal'),
                               tf.keras.layers.BatchNormalization(),
                               tf.keras.layers.Activation('relu'),
                               tf.keras.layers.Dense(10, kernel_initializer='he_normal', activation='softmax', name='predictions')
                               ])'''