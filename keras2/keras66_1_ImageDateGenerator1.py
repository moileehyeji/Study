# ImageDataGenerator
# 이미지 전처리

import numpy as np  
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 이미지 가져오기전 ImageDataGenerator 생성
# (x, y) 튜플을 만들어내는 DirectoryIterator 여기서 
# x는 (배치 크기, *표적 크기, 채널)의 형태의 이미지 배치로 구성된 numpy 배열이고 
# y는 그에 대응하는 라벨로 이루어진 numpy 배열입니다.
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
    batch_size = 5,
    class_mode = 'binary'           # 분류 방식 지정 =  이진 라벨이 반환 0,1 
)
xy_test = test_dategen.flow_from_directory(
    '../data/Image/brain/test',
    target_size = (150,150),        # 패치 이미지 크기를 지정합니다. 폴더에 있는 원본 이미지 크기가 다르더라도 target_size에 지정된 크기로 자동 조절(150,150)
    batch_size = 5,
    class_mode = 'binary'           # 분류 방식 지정  =  이진 라벨이 반환 0,1 
)

# Found 160 images belonging to 2 classes.
# Found 120 images belonging to 2 classes.

print(xy_train)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x00000242853A8550>

print(xy_train[0])           # x, y 모두 가지고 있음
print(xy_train[0][0])        # x
print(xy_train[0][0].shape)  # x (10, 150, 150, 3)   batch_size=10
print(xy_train[0][1])        # y (10, 150, 150, 3) [0. 1. 1. 0. 1. 1. 0. 0. 0. 1.]   batch_size=10
print(xy_train[0][1].shape)  # y (10,)   batch_size=10
# print(xy_train[15][1])     # y [1. 0. 0. 0. 1. 0. 0. 0. 1. 0.]
# xy_train[15][1]-> (train160장) / (batch_size10) = 16(0~15)
# batch_size = 260 데이터 크기를 넘어가는 경우 자동 (160,-)

