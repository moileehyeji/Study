""" # VGG16으로 만들어 봐!

from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# 1.데이터
path = 'C:/data/Image/gender'
img_hyeji = load_img(f'{path}/hyeji.jpg', target_size=(224,224))

# print(img_hyeji) # <PIL.Image.Image image mode=RGB size=224x224 at 0x1A27E3B8070>

# plt.imshow(img_hyeji)
# plt.show()
 
# 이미지 수치화
arr_hyeji = img_to_array(img_hyeji)
print(arr_hyeji)
print(type(arr_hyeji))    #<class 'numpy.ndarray'>
print(arr_hyeji.shape)    #(64, 64, 3)


#RGB -> BGR
from tensorflow.keras.applications.vgg16 import preprocess_input
# VGG16 전이모델에 맞춰 preprocessing해준다
arr_hyeji = preprocess_input(arr_hyeji)
# print(type(arr_dog))    #<class 'numpy.ndarray'>
# print(arr_dog.shape)    #(224, 224, 3) -> VGG16 기본값?

# 이미지이므로 4차원으로 훈련시켜야 한다
# 4개 이미지 합하면? (4, 224, 224, 3) 4차원
# np.stack: 새 축을 따라 일련의 배열을 결합(합치려는 배열들의 shape이 전부 동일해야함)
arr_input = arr_hyeji.reshape(1, 224, 224, 3)
print(arr_input.shape)  #(1, 224, 224, 3)

# 2.모델구성
# 우리 훈련안시키고 결과만 볼거야
model = VGG16()
results = model.predict(arr_input)

print(results)  #수치화된 이미지 결과
print('results.shape : ', results.shape)    #results.shape :  (4, 1000), 1000: ImageNet에서 분류할수 있는 category 수

#results(수치화된 이미지 결과) 확인하기
# decode_predictions : ImageNet 모델의 예측을 디코딩
# 반환 : 최고 수준의 예측 튜플 목록. 
#        일괄 입력에서 샘플 당 하나의 튜플 목록. (class_name, class_description, score)
# pred배열의 모양이 잘못된 경우 (2D 여야 함).
from tensorflow.keras.applications.vgg16 import decode_predictions

decode_results = decode_predictions(results)
print('--------------------------------------------------------')
print('results[0] : ', decode_results[0])
print('--------------------------------------------------------') """

import numpy as np
import os 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.applications import VGG16, InceptionV3, Xception

# train, test npy load
x_train = np.load('../data/Image/gender/npy/keras67_1_train_x.npy')
y_train = np.load('../data/Image/gender/npy/keras67_1_train_y.npy')
x_val = np.load('../data/Image/gender/npy/keras67_1_val_x.npy')
y_val = np.load('../data/Image/gender/npy/keras67_1_val_y.npy')

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2, shuffle = True, random_state=66)

print(x_train.shape)    #(972, 32, 32, 3)
print(x_test.shape)     #(244, 32, 32, 3)
print(x_val.shape)      #(520, 32, 32, 3)
print(y_train.shape)    #(972, 1)
print(y_test.shape)     #(244, 1)
print(y_val.shape)      #(520, 1)

# 2. 모델
def build_model(drop=0.5, optimizer=Adam, filters=100, kernel_size=2, learning_rate=0.1):
    vgg16 = InceptionV3(weights='imagenet',       #imagenet데이터로 저장된 가중치
            include_top=False,              #False일때 input_shape 변경가능 
            input_shape = (96,96,3))
    vgg16.trainable = False # 동결(freezen)
    
    model = Sequential()
    model.add(UpSampling2D(size = (3,3)))  
    model.add(vgg16)
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1,activation='sigmoid'))

    model.compile( optimizer=Nadam(lr=0.005), loss='binary_crossentropy', metrics=['acc'])
    return model

def callbacks():
    modelpath ='../data/modelcheckpoint/k67_2_{epoch:2d}_{val_loss:.4f}.hdf5'
    er = EarlyStopping(monitor = 'val_loss',patience=50)
    mo = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath,save_best_only=True)
    lr = ReduceLROnPlateau(monitor = 'val_loss', patience=25,factor=0.5 ,verbose=1)
    return er,mo,lr

er,mo,lr = callbacks() 

model = build_model()
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),batch_size=20, epochs=500, callbacks = [er,lr])
model.save('../data/h5/keras81_male_female.h5')

''' # 모델저장
hdf5_file = "../data/h5/keras81_male_female.h5"
if os.path.exists(hdf5_file):
    # 기존에 학습된 모델 불러들이기
    model.load_weights(hdf5_file)
else:
    # 학습한 모델이 없으면 파일로 저장
    history =  model.fit(x_train, y_train, validation_data=(x_val, y_val),batch_size=8, epochs=500, callbacks = [er,lr])
    model.save('../data/h5/keras81_male_female.h5') '''

loss, acc = model.evaluate(x_test, y_test, batch_size=8)
print('loss, acc : ', loss, acc)

'''
CNN:
loss, acc :  0.6761801242828369 0.5819672346115112

VGG:
loss, acc :  0.6985849142074585 0.4959016442298889
loss, acc :  0.7016910910606384 0.5163934230804443
loss, acc :  0.7362603545188904 0.5286885499954224
loss, acc :  0.7490838766098022 0.5327869057655334
loss, acc :  0.7125701904296875 0.5450819730758667

InceptionV3:
loss, acc :  1.490882396697998 0.5368852615356445
'''
