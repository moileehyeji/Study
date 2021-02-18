## 꼭 뿌신다 내가

import pandas as pd
import os


csv_train = pd.read_csv('C:/data/dacon_mnist/train.csv')
csv_test = pd.read_csv('C:/data/dacon_mnist/test.csv')

csv_train.head()
'''
# train 이미지들과 test 이미지들을 저장해놓을 폴더를 생성합니다.
os.mkdir('C:/data/dacon_mnist/image_train')
os.mkdir('C:/data/dacon_mnist/image_train/0')
os.mkdir('C:/data/dacon_mnist/image_train/1')
os.mkdir('C:/data/dacon_mnist/image_train/2')
os.mkdir('C:/data/dacon_mnist/image_train/3')
os.mkdir('C:/data/dacon_mnist/image_train/4')
os.mkdir('C:/data/dacon_mnist/image_train/5')
os.mkdir('C:/data/dacon_mnist/image_train/6')
os.mkdir('C:/data/dacon_mnist/image_train/7')
os.mkdir('C:/data/dacon_mnist/image_train/8')
os.mkdir('C:/data/dacon_mnist/image_train/9')
os.mkdir('C:/data/dacon_mnist/image_test')
'''
import cv2

for idx in range(len(csv_train)) :
    img = csv_train.loc[idx, '0':].values.reshape(28, 28).astype(int)
    digit = csv_train.loc[idx, 'digit']
    cv2.imwrite(f'C:/data/dacon_mnist/image_train/{digit}/{csv_train["id"][idx]}.png', img)

for idx in range(len(csv_test)) :
    img = csv_test.loc[idx, '0':].values.reshape(28, 28).astype(int)
    cv2.imwrite(f'C:/data/dacon_mnist/image_test/{csv_test["id"][idx]}.png', img)

import tensorflow as tf

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
                               ])

model_1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2,
                             rotation_range=10,
                             width_shift_range=0.1, 
                             height_shift_range=0.1)

train_generator = datagen.flow_from_directory(
          'C:/data/dacon_mnist/image_train', target_size=(224,224), color_mode='grayscale', 
             class_mode='categorical', subset='training')
val_generator = datagen.flow_from_directory(
        'C:/data/dacon_mnist/image_train', target_size=(224,224), color_mode='grayscale', 
           class_mode='categorical', subset='validation')


checkpoint_1 = tf.keras.callbacks.ModelCheckpoint(f'C:/data/modelCheckpoint/0205_1_model_1.h5', 
                             monitor='val_accuracy', save_best_only=True, verbose=1)
checkpoint_2 = tf.keras.callbacks.ModelCheckpoint(f'C:/data/modelCheckpoint/0205_1_model_2.h5', 
                             monitor='val_accuracy', save_best_only=True, verbose=1)
checkpoint_3 = tf.keras.callbacks.ModelCheckpoint(f'C:/data/modelCheckpoint/0205_1_model_3.h5', 
                             monitor='val_accuracy', save_best_only=True, verbose=1)

model_1.fit_generator(train_generator, epochs=300, validation_data=val_generator, callbacks=[checkpoint_1])
model_2.fit_generator(train_generator, epochs=300, validation_data=val_generator, callbacks=[checkpoint_2])
model_3.fit_generator(train_generator, epochs=300, validation_data=val_generator, callbacks=[checkpoint_3])


'''
import os
import random
import pandas as pd
import numpy as np
import os
from glob import glob
from keras.preprocessing import image as krs_image
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import PIL.Image as pilimg
import cv2
from collections import Counter

model_1 = tf.keras.models.load_model('C:/data/modelCheckpoint/0203_1_model_1.h5', compile=False)
model_2 = tf.keras.models.load_model('C:/data/modelCheckpoint/0203_1_model_2.h5', compile=False)
model_3 = tf.keras.models.load_model('C:/data/modelCheckpoint/0203_1_model_3.h5', compile=False)

submission = pd.read_csv('C:/data/dacon_mnist/submission.csv')
pred1=[]
pred2=[]
pred3=[]
for i in range(2049,22529):
    picture_path = "C:/data/dacon_mnist/image_test/none/{}.png".format(i)
    pix = cv2.imread(picture_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(pix,(224,224))
    image = image.reshape(224,224,1).astype('float32')/255.0
    y_pred1 = model_1.predict(np.array([image]))
    y_pred2 = model_2.predict(np.array([image]))
    y_pred3 = model_3.predict(np.array([image]))
    y_pred1 = np.argmax(y_pred1,axis=1)
    y_pred2 = np.argmax(y_pred2,axis=1)
    y_pred3 = np.argmax(y_pred3,axis=1)
    pred1.append(y_pred1[0])
    pred2.append(y_pred2[0])
    pred3.append(y_pred3[0])
    if (i%100)==0:
        print("{}% 진행중".format((i)/(22529)))

submission['predict_1']=pred1
submission['predict_2']=pred2
submission['predict_3']=pred3

submission.to_csv('C:/data/dacon_mnist/answer/0205_1_mnist.csv',index=False)
submission = pd.read_csv('C:/data/dacon_mnist/answer/0205_1_mnist.csv')

for i in range(len(submission)) :
    predicts = submission.loc[i, ['predict_1','predict_2','predict_3']]
    submission.at[i, "digit"] = Counter(predicts).most_common(n=1)[0][0]

submission = submission[['id', 'digit']]
submission.to_csv('C:/data/dacon_mnist/answer/0205_1_mnist.csv', index=False)
'''