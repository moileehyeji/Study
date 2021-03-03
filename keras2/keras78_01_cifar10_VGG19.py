# 실습
# cifar10으로 vgg16 넣어서 만들 것
# 결과치에 대한 기본값과 비교

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import VGG19
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
from tensorflow.keras.applications.vgg19 import preprocess_input

# 1.데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, shuffle = True, random_state = 66)

# print(x_train.shape)    #(40000, 32, 32, 3)
# print(x_test.shape)     #(10000, 32, 32, 3)
# print(x_val.shape)      #(10000, 32, 32, 3)
# print(y_train.shape)    #(40000, 1)
# print(y_test.shape)     #(10000, 1)
# print(y_val.shape)      #(10000, 1)

# 전이학습 모델에 맞게 기본으로 필요한 전처리
# mode = 'caffe'(기본값) : 이미지를 RGB에서 BGR로 변환한 다음 스케일링없이 ImageNet데이터셋과 관련해 각 색상채널의 중심을 제로화
#        'tf' : 샘플단위로 -1rhk1사이의 픽셀크기를 조정
#        'torch': 0과 1사이의 픽셀크기를 조정한 다음 ImageNet데이터셋과 관련하여 각 채널을 정규화
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)
x_val = preprocess_input(x_val)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

# 2. 모델
vgg19 = VGG19(weights='imagenet',
            include_top=False,
            input_shape=(x_train.shape[1:]))
vgg19.trainable = False

model = Sequential()
model.add(vgg19)
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()
""" 
Total params: 20,061,898
Trainable params: 37,514
Non-trainable params: 20,024,384
"""
def callbacks():
    modelpath ='../data/modelcheckpoint/k67_3_{epoch:2d}_{val_loss:.4f}.hdf5'
    er = EarlyStopping(monitor = 'val_loss',patience=5)
    mo = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath,save_best_only=True)
    lr = ReduceLROnPlateau(monitor = 'val_loss', patience=3,factor=0.3 ,verbose=1)
    return er,mo,lr
er,mo,lr = callbacks()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer=Nadam(learning_rate=0.005), metrics='acc')
model.fit(x_train, y_train, epochs=100, batch_size=35, validation_data=(x_val, y_val), callbacks=[er, lr])

# 4. 평가
loss = model.evaluate(x_test, y_test, batch_size=35)
print('loss, acc : ', loss)


'''
keras67_3_cifar10
IDG + CNN:
loss, acc :  [0.9830985069274902, 0.6495000123977661]
===========================================================
1. vgg16
loss, acc :  [1.1218377351760864, 0.6552000045776367]

2. vgg19
loss, acc :  [1.1227667331695557, 0.6488999724388123]
'''

