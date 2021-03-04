# 실습
# 남녀구별
# ImageDataGenerator flow_from_directory, fit_generator


import numpy as np  
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


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
    fill_mode='nearest',         # 빈자리 주변 유사수치로 채움(=0이면 0으로 채움)
    validation_split=0.3
)
test_dategen = ImageDataGenerator(rescale=1./255) # test 이미지는 증폭XXX, 0-1사이값으로 정규화

train_data = 1216
val_data = 520
batch_size1 = 1216
batch_size2 = 520
xy_train = train_datagen.flow_from_directory(
    '../data/Image/gender',
    target_size = (96,96),                
    batch_size = batch_size1,
    class_mode = 'binary',               
    subset = 'training'
)
xy_val = train_datagen.flow_from_directory(
    '../data/Image/gender',
    target_size = (96,96),               
    batch_size = batch_size2,
    class_mode = 'binary',                
    subset = 'validation'
)
# Found 1216 images belonging to 3 classes.
# Found 520 images belonging to 3 classes.

# print(train_generator[0])     # x, y 모두 가지고 있음
# print(train_generator[0][0])  # x
print(xy_train[0][0].shape)     # x (463, 64, 64, 3)
# print(train_generator[0][1])  # y 
print(xy_val[0][1].shape)       # y (32,)

# train, test npy 저장
np.save('../data/Image/gender/npy/keras67_1_96_train_x.npy', arr=xy_train[0][0])
np.save('../data/Image/gender/npy/keras67_1_96_train_y.npy', arr=xy_train[0][1])
np.save('../data/Image/gender/npy/keras67_1_96_val_x.npy', arr=xy_val[0][0])
np.save('../data/Image/gender/npy/keras67_1_96_val_y.npy', arr=xy_val[0][1])


'''
# 2. 모델
def build_model(drop=0.5, optimizer=Adam, filters=100, kernel_size=2, learning_rate=0.1):
    model = Sequential()
    model.add(Conv2D(16,(3,3),activation='relu', input_shape=(32, 32, 3),padding='same'))
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
    model.compile( optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
    return model


def callbacks():
    modelpath ='../data/modelcheckpoint/k61_4_{epoch:2d}_{val_loss:.4f}.hdf5'
    er = EarlyStopping(monitor = 'val_loss',patience=5)
    mo = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath,save_best_only=True)
    lr = ReduceLROnPlateau(monitor = 'val_loss', patience=3,factor=0.3 ,verbose=1)
    return er,mo,lr

er,mo,lr = callbacks() 

model = build_model()
history = model.fit_generator(generator=xy_train, validation_data=xy_val, steps_per_epoch=int(train_data/batch_size1), 
                                    validation_steps=int(val_data/batch_size2), epochs=500, callbacks = [er,lr])

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

print('acc : ', np.mean(acc))
print('val_acc : ', np.mean(val_acc))
'''
'''
# 시각화 할 것
# val_loss와 loss의 간격이 좁을수록 좋은 성능
import matplotlib.pyplot as plt
plt.plot(loss)
plt.plot(val_loss)
plt.plot(acc)
plt.plot(val_acc)
plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epochs')
plt.legend(['train loss', 'val loss', 'train acc', 'val acc'])
plt.show()
'''



'''
acc :  0.515537829041481
val_acc :  0.5152538452148437
'''