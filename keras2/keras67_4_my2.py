# 나를 찍어서 여자인지 남자인지에 대해
# predict
# 여자라면 그 acc까지

import numpy as np  
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# # train, test npy load
# x_train = np.load('../data/Image/gender/npy/keras67_4_train_x.npy')
# y_train = np.load('../data/Image/gender/npy/keras67_4_train_y.npy')
# x_val = np.load('../data/Image/gender/npy/keras67_4_val_x.npy')
# y_val = np.load('../data/Image/gender/npy/keras67_4_val_y.npy')

# train, test npy load
x_train = np.load('../data/Image/gender/npy/keras67_1_train_x.npy')
y_train = np.load('../data/Image/gender/npy/keras67_1_train_y.npy')
x_val = np.load('../data/Image/gender/npy/keras67_1_val_x.npy')
y_val = np.load('../data/Image/gender/npy/keras67_1_val_y.npy')


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2, shuffle = True, random_state=66)

print(x_train.shape)   
print(x_test.shape)    
print(x_val.shape)      
print(y_train.shape)    
print(y_test.shape)     
print(y_val.shape)      
'''
(1180, 100, 100, 3)
(296, 100, 100, 3)
(260, 100, 100, 3)
(1180,)
(296,)
(260,)
'''

# 2. 모델
def build_model(drop=0.5, optimizer=Adam, filters=100, kernel_size=2, learning_rate=0.1):
    
    model = Sequential()
    model.add(Conv2D(16,(3,3),activation='relu', input_shape=(x_train.shape[1], x_train.shape[2], 3),padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(5,5),activation='relu',padding='same')) 
    model.add(BatchNormalization())
    model.add(Conv2D(128,(5,5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128,(5,5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3,3)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128,(5,5),activation='relu',padding='same')) 
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
    '''
    model = Sequential([
    # 1st conv
    Conv2D(96, (3,3),strides=(4,4), padding='same', activation='relu', input_shape=(x_train.shape[1], x_train.shape[2], 3)),
    BatchNormalization(),
    MaxPooling2D(2, strides=(2,2)),
        # 2nd conv
    Conv2D(256, (3,3),strides=(1,1), activation='relu',padding="same"),
    BatchNormalization(),
        # 3rd conv
    Conv2D(256, (3, 3), strides=(1, 1), activation='relu',padding="same"),
    BatchNormalization(),
    MaxPooling2D(2, strides=(2, 2)),
        # To Flatten layer
    Flatten(),
    Dropout(0.5),
        #To FC layer 1
    Dense(30, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')
    ])'''
    
    # 컴파일
    model.compile( optimizer=Adam(lr=0.03), loss='binary_crossentropy', metrics=['acc'])
    return model

def callbacks():
    modelpath ='../data/modelcheckpoint/k67_4_{epoch:2d}_{val_loss:.4f}.hdf5'
    er = EarlyStopping(monitor = 'val_loss',patience=5)
    mo = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath,save_best_only=True)
    lr = ReduceLROnPlateau(monitor = 'val_loss', patience=3,factor=0.2 ,verbose=1)
    return er,mo,lr

#훈련
er,mo,lr = callbacks() 
model = build_model()
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),batch_size=8, epochs=500, callbacks = [er,lr])

# 모델저장
model.save('../data/h5/k67_4_model.h5')
model.save_weights('../data/h5/k67_4_weights.h5')

#평가
loss, acc = model.evaluate(x_test, y_test, batch_size=8)
print('loss, acc : ', loss, acc)

#예측)
# y_pred = model.predict(x_test)
# print(y_pred)

'''
xy_train = test_dategen.flow(img, batch_size = 520)

y_pred = model.predict(img)
y_pred = np.where(y_pred>0.5, 'male', 'female')
print(y_pred)
'''

""" import numpy as np
from keras.preprocessing import image

# predicting images
path = "C:/data/image/ma_female/female/final_1000.jpg"
img = image.load_img(path, target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=1)
print(classes[0])
if classes[0]>0.5:
    print("is a man")
else:
    print( " is a female")
plt.imshow(img)
plt.show() """

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
loss, acc :  0.6624506115913391 0.5819672346115112
'''