# ImageDataGenerator
# 이미지 전처리

import numpy as np  
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 이미지 가져오기전 ImageDataGenerator 생성
train_datagen = ImageDataGenerator(
    rescale= 1./255,            # 0-1사이값으로 정규화
    horizontal_flip=True,       # 수평뒤집음
    vertical_flip=True,         # 수직뒤집음
    width_shift_range=0.1,      # 수평이동
    height_shift_range=0.1,     # 수직이동
    rotation_range=5,           # 무작위회전
    zoom_range=1.2,             # 임의확대, 축소
    shear_range=0.7,            # 층밀리기의 강도
    fill_mode='nearest'         # 빈자리 주변 유사수치로 채움(=0이면 0으로 채움)
)
test_dategen = ImageDataGenerator(rescale=1./255) # test 이미지는 증폭XXX, 0-1사이값으로 정규화



# train, test_generator 이미지 가져오기(증폭전)
# flow : 이미 수치화된 이미지파일에 라벨링 (ex)mnist)
# flow_from_directory : 폴더자체(폴더내 전체이미지)에 라벨링 할 수 있다(ad, normal)
#                       이미지데이터의 fit과 같은 개념

# C:\data\Image\brain\train\ad      -> x, (80,150,150,1or3) , 0
# C:\data\Image\brain\train\normal  -> y, (80,), 1
# C:\data\Image\brain\test\ad      -> x, (60,150,150,1or3) , 0
# C:\data\Image\brain\test\normal  -> y, (60,), 1
xy_train = train_datagen.flow_from_directory(
    '../data/Image/brain/train',
    target_size = (150,150),        # 패치 이미지 크기를 지정합니다. 폴더에 있는 원본 이미지 크기가 다르더라도 target_size에 지정된 크기로 자동 조절(150,150)
    batch_size = 160,
    class_mode = 'binary'           # 분류 방식 지정 =  이진 라벨이 반환 0,1 
)
xy_test = test_dategen.flow_from_directory(
    '../data/Image/brain/test',
    target_size = (150,150),        # 패치 이미지 크기를 지정합니다. 폴더에 있는 원본 이미지 크기가 다르더라도 target_size에 지정된 크기로 자동 조절(150,150)
    batch_size = 120,
    class_mode = 'binary'           # 분류 방식 지정  =  이진 라벨이 반환 0,1 
)

# Found 160 images belonging to 2 classes.
# Found 120 images belonging to 2 classes.

print(xy_train)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x00000242853A8550>

print(xy_train[0])           # x, y 모두 가지고 있음
print(xy_train[0][0])        # x
print('============================================================')
print(xy_train[0][0].shape)  # x (160, 150, 150, 3)   batch_size=160
print(xy_train[0][1])        # y 
print(xy_train[0][1].shape)  # y (160,)   batch_size=160



# train, test npy 저장
np.save('../data/Image/brain/npy/keras66_3_train_x.npy', arr=xy_train[0][0])
np.save('../data/Image/brain/npy/keras66_3_train_y.npy', arr=xy_train[0][1])
np.save('../data/Image/brain/npy/keras66_3_test_x.npy', arr=xy_test[0][0])
np.save('../data/Image/brain/npy/keras66_3_test_y.npy', arr=xy_test[0][1])

x_train = np.load('../data/Image/brain/npy/keras66_3_train_x.npy')
y_train = np.load('../data/Image/brain/npy/keras66_3_train_y.npy')
x_test = np.load('../data/Image/brain/npy/keras66_3_test_x.npy')
y_test = np.load('../data/Image/brain/npy/keras66_3_test_y.npy')

print(x_train.shape, y_train.shape) # (160, 150, 150, 3) (160,)
print(x_test.shape, y_test.shape)   # (120, 150, 150, 3) (120,)


