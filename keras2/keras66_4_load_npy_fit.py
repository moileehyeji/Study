# ImageDataGenerator
# 이미지 전처리

import numpy as np  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adamax
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

x_train = np.load('../data/Image/brain/npy/keras66_3_train_x.npy')
y_train = np.load('../data/Image/brain/npy/keras66_3_train_y.npy')
x_test = np.load('../data/Image/brain/npy/keras66_3_test_x.npy')
y_test = np.load('../data/Image/brain/npy/keras66_3_test_y.npy')

print(x_train.shape, y_train.shape) # (160, 150, 150, 3) (160,)
print(x_test.shape, y_test.shape)   # (120, 150, 150, 3) (120,)

# 실습
# 모델 만들기

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, shuffle = True, random_state=66)

# 2. 모델
def build_model(drop=0.5, optimizer=Adam, filters=100, kernel_size=2, learning_rate=0.1):
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
    model.compile(optimizer=Adam(lr=0.1), loss='binary_crossentropy', metrics=['acc'])
    return model

def callbacks():
    modelpath ='../data/modelcheckpoint/k66_4_{epoch:2d}_{val_loss:.4f}.hdf5'
    er = EarlyStopping(monitor = 'val_loss',patience=5)
    mo = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath,save_best_only=True)
    lr = ReduceLROnPlateau(monitor = 'val_loss', patience=3,factor=0.5 ,verbose=1)
    return er,mo,lr

er,mo,lr = callbacks() 

model = build_model()
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),batch_size=10, epochs=500, callbacks = [er,lr])

loss, acc = model.evaluate(x_test, y_test, batch_size=10)
print('loss, acc : ', loss, acc)

# loss, acc :  0.40351417660713196 0.8166666626930237



