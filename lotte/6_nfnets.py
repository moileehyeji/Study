# NFNetF3 아직 Image Net으로 훈련 안됨


from tensorflow.keras.datasets import cifar10
from nfnets_keras import NFNetF3
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

model = NFNetF3(include_top = True, num_classes = 10)

#3. 모델컴파일, 훈련
model.compile('adam', 'categorical_crossentropy')
model.fit(x_train, y_train)

#4.모델평가, 예측
loss=model.evaluate(x_test,y_test,batch_size=1) #compile 'mse'와 연결
print('loss: ',loss)






''' import numpy as np
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation, Dropout
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import VGG19, MobileNet, ResNet101
from nfnets_keras import NFNetF3

#데이터 지정 및 전처리
x = np.load("C:/Study/lotte/data/npy/128_project_x.npy",allow_pickle=True)
x_pred = np.load('C:/Study/lotte/data/npy/128_test.npy',allow_pickle=True)
y = np.load("C:/Study/lotte/data/npy/128_project_y.npy",allow_pickle=True)

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

x_train, x_valid, y_train, y_valid = train_test_split(x,y, train_size = 0.9, shuffle = True, random_state=66)

train_generator = idg.flow(x_train,y_train,batch_size=64, seed = 2048)
# seed => random_state
valid_generator = idg2.flow(x_valid,y_valid)
# test_generator = idg2.flow(x_pred)

mc = ModelCheckpoint('C:/Study/lotte/data/h5/6_nfnets.h5',save_best_only=True, verbose=1)
# efficientnet = EfficientNetB4(include_top=False,weights='imagenet',input_shape=x_train.shape[1:])
nf = NFNetF3(include_top=False,weights='imagenet',input_shape=x_train.shape[1:])
nf.trainable = True
a = nf.output
a = GlobalAveragePooling2D() (a)
a = Flatten() (a)
a = Dense(4048, activation= 'relu') (a)
a = Dropout(0.2) (a)
a = Dense(1000, activation= 'softmax') (a)

model = Model(inputs = nf.input, outputs = a)

early_stopping = EarlyStopping(patience= 20)
lr = ReduceLROnPlateau(patience= 10, factor=0.5)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
learning_history = model.fit_generator(train_generator,epochs=200, steps_per_epoch= len(x_train) / 64,
    validation_data=valid_generator, callbacks=[early_stopping,lr,mc])

# predict
model.load_weights('C:/Study/lotte/data/h5/6_nfnets.h5')
result = model.predict(x_pred,verbose=True)


# 제출생성
sub = pd.read_csv('C:/Study/lotte/data/sample.csv')
sub['prediction'] = np.argmax(result,axis = 1)
sub.to_csv('C:/Study/lotte/data/6.csv',index=False) '''