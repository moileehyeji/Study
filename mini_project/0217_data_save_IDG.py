import numpy as np  
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale= 1./255,            # 0-1사이값으로 정규화
    horizontal_flip=True,       # 수평뒤집음
    # vertical_flip=True,         # 수직뒤집음
    width_shift_range=0.1,      # 수평이동
    height_shift_range=0.1,     # 수직이동
    rotation_range=5,           # 무작위회전 
    zoom_range=1.2,             # 임의확대, 축소
    # shear_range=0.7,            # 층밀리기의 강도
    fill_mode='nearest',         # 빈자리 주변 유사수치로 채움(=0이면 0으로 채움)
    validation_split=0.25
    
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
    './project/data/img',
    target_size = (150,150),        # 패치 이미지 크기를 지정합니다. 폴더에 있는 원본 이미지 크기가 다르더라도 target_size에 지정된 크기로 자동 조절(150,150)
    batch_size = 50000,
    class_mode = 'categorical'   ,        # 분류 방식 지정 =  이진 라벨이 반환 0,1 
    subset='training'
)
xy_val = train_datagen.flow_from_directory(
    './project/data/img',
    target_size = (150,150),        # 패치 이미지 크기를 지정합니다. 폴더에 있는 원본 이미지 크기가 다르더라도 target_size에 지정된 크기로 자동 조절(150,150)
    batch_size = 50000,
    class_mode = 'categorical' ,          # 분류 방식 지정  =  이진 라벨이 반환 0,1 
    subset='validation'
)

# train, test npy 저장
np.save('./project/data/npy/0217_IDG_train_x.npy', arr=xy_train[0][0])
np.save('./project/data/npy/0217_IDG_train_y.npy', arr=xy_train[0][1])
np.save('./project/data/npy/0217_IDG_1_val_x.npy', arr=xy_val[0][0])
np.save('./project/data/npy/0217_IDG_1_val_y.npy', arr=xy_val[0][1])
print('----------> save')

# Found 3163 images belonging to 7 classes.
# Found 1050 images belonging to 7 classes.

# print(xy_train)

# print(xy_train[0])           # x, y 모두 가지고 있음
# print(xy_train[0][0])        # x
# print(xy_train[0][0].shape)  # x (10, 150, 150, 3)
# print(xy_train[0][1])        # y 
# print(xy_train[0][1].shape)  # y (10, 7)
 
