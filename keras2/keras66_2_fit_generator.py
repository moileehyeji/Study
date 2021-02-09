# ImageDataGenerator
# 이미지 전처리

import numpy as np  
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam

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
train_data = 160
batch_size = 5
xy_train = train_datagen.flow_from_directory(
    '../data/Image/brain/train',
    target_size = (150,150),        # 패치 이미지 크기를 지정합니다. 폴더에 있는 원본 이미지 크기가 다르더라도 target_size에 지정된 크기로 자동 조절(150,150)
    batch_size = batch_size,
    class_mode = 'binary'           # 분류 방식 지정 =  이진 라벨이 반환 0,1 
)
xy_test = test_dategen.flow_from_directory(
    '../data/Image/brain/test',
    target_size = (150,150),        # 패치 이미지 크기를 지정합니다. 폴더에 있는 원본 이미지 크기가 다르더라도 target_size에 지정된 크기로 자동 조절(150,150)
    batch_size = batch_size,
    class_mode = 'binary'           # 분류 방식 지정  =  이진 라벨이 반환 0,1 
)

# Found 160 images belonging to 2 classes.
# Found 120 images belonging to 2 classes.

model = Sequential()
model.add(Conv2D(16,(3,3),activation='relu', input_shape=(150, 150, 3),padding='same'))
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

model.add(Dense(1,activation='sigmoid'))

    
model.compile(loss = 'binary_crossentropy', optimizer=Adamax(learning_rate=0.005), metrics='acc')

##############
def callbacks():
    modelpath ='../data/modelcheckpoint/k66_2_{epoch:2d}_{val_loss:.4f}.hdf5'
    er = EarlyStopping(monitor = 'val_loss',patience=50)
    mo = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath,save_best_only=True)
    lr = ReduceLROnPlateau(monitor = 'val_loss',patience=2, factor=0.3)
    return er,mo,lr

er,mo,lr = callbacks() 
##############

# fit_generator : xy데이터 
# steps_per_epoch : (train 데이터 갯수 160개) / (batch_size 5) = 32 
# validation_steps : 한 epoch 종료 시 마다 검증할 때 사용되는 검증 스텝 수를 지정
history = model.fit_generator(
    xy_train, steps_per_epoch=(train_data/batch_size), epochs=500, 
    validation_data=xy_test, validation_steps=4, 
    callbacks=[er,mo,lr]
)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']


# 시각화 할 것
# val_loss와 loss의 간격이 좁을수록 좋은 성능
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers.normalization import BatchNormalization
plt.plot(loss)
plt.plot(val_loss)
plt.plot(acc)
plt.plot(val_acc)
plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epochs')
plt.legend(['train loss', 'val loss', 'train acc', 'val acc'])
plt.show()

print('acc : ', np.mean(acc))
print('val_acc : ', np.mean(val_acc))
