# 실습
# 남녀구별
# ImageDataGenerator flow_from_directory, fit

import numpy as np  
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

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
    model.compile( optimizer=Adam(lr=0.1), loss='binary_crossentropy', metrics=['acc'])
    return model

def callbacks():
    modelpath ='../data/modelcheckpoint/k67_2_{epoch:2d}_{val_loss:.4f}.hdf5'
    er = EarlyStopping(monitor = 'val_loss',patience=5)
    mo = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath,save_best_only=True)
    lr = ReduceLROnPlateau(monitor = 'val_loss', patience=3,factor=0.5 ,verbose=1)
    return er,mo,lr

er,mo,lr = callbacks() 

model = build_model()
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),batch_size=8, epochs=500, callbacks = [er,lr])

loss, acc = model.evaluate(x_test, y_test, batch_size=8)
print('loss, acc : ', loss, acc)

'''
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

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
loss, acc :  0.6761801242828369 0.5819672346115112
'''