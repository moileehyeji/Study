# 실습
# cifar10으로 vgg16 넣어서 만들 것
# 결과치에 대한 기본값과 비교

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import EfficientNetB0
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, UpSampling2D, Dropout
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
efficientnetb0= EfficientNetB0(weights='imagenet',
            include_top=False,
            input_shape=(224,224,3))
efficientnetb0.trainable = False

# ValueError: When setting `include_top=True` and loading `imagenet` weights, `input_shape` should be (224, 224, 3).
model = Sequential()
model.add(UpSampling2D(size = (7,7)))   #(96,96,3)
model.add(efficientnetb0)
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# model.summary()

def callbacks():
    modelpath ='../data/modelcheckpoint/k67_3_{epoch:2d}_{val_loss:.4f}.hdf5'
    er = EarlyStopping(monitor = 'val_loss',patience=5)
    mo = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath,save_best_only=True)
    lr = ReduceLROnPlateau(monitor = 'val_loss', patience=3,factor=0.3 ,verbose=1)
    return er,mo,lr
er,mo,lr = callbacks()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer=Nadam(learning_rate=0.1), metrics='acc')
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

3. Xception
loss, acc :  [0.8433055877685547, 0.7750999927520752]

4. ResNet50
loss, acc :  [1.1748799085617065, 0.656499981880188]

5. ResNet101
loss, acc :  [1.1258513927459717, 0.6582000255584717]

6. inceptionv3
loss, acc :  [2.094517469406128, 0.18410000205039978]

7. inceptionresnetv2
loss, acc :  [44.1339, acc: 0.1613]

8. DenseNet121
loss, acc :  [2.302617073059082, 0.10000000149011612]

9. mobilenetv2
loss, acc :  [2.3026063442230225, 0.10000000149011612]

10. NasNetMobile
loss, acc :  [2.3026342391967773, 0.10000000149011612]

11. EfficientNetB0
loss, acc :  [2.303347110748291, 0.10000000149011612]
'''
