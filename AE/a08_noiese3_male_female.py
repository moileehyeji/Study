# [실습] K67_1 남자여자 noise넣어서
# 기미 주근깨 여드름을 제거하시오

import numpy as np  
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# train, test npy load
x_train = np.load('../data/Image/gender/npy/keras67_1_a08_train_x.npy')

x_train, x_test = train_test_split(x_train, test_size = 0.2, shuffle = True, random_state=66)

x_train_out = x_train.reshape(-1,40*40)

print(x_train.shape, x_test.shape, x_train_out.shape)    #(1388, 40, 40, 1) (348, 40, 40, 1) (1736, 1024)

# 노이즈 만들기
x_train_noised = x_train + np.random.normal(0, 0.1, size = x_train.shape)   #0~0.1값을 랜덤하게 더해서 노이즈 만들기 (0~1.1분포)
x_test_noised = x_test + np.random.normal(0, 0.1, size = x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)  # (0~1.1분포) --> 1보다 크면 1로 고정
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)
# 0~1  --> 0~1 같은 범위이지만 노이즈가 생김



from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, UpSampling2D, BatchNormalization

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, kernel_size=2, padding='same', strides=1, input_shape = (40,40,1), activation='relu'))
    model.add(MaxPooling2D(2))
    model.add(BatchNormalization())
    model.add(Conv2D(128, 2, 1, activation='relu'))
    model.add(UpSampling2D(2))
    model.add(Flatten())
    # model.add(Dense(40, activation='relu'))
    # model.add(Dense(140, activation='relu'))
    # model.add(Dense(154, activation='relu'))
    model.add(Dense(units=40*40, activation='sigmoid'))
    return model

model = autoencoder(hidden_layer_size=154)
# 95% PCA수치 154 가장 안정적인 수치로 이미지 복원이 되는지 확인하자

model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['acc'])

model.fit(x_train_noised, x_train_out, epochs=10, batch_size=256)

output = model.predict(x_test_noised)


import matplotlib.pyplot as plt 
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10),
    (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3,5,figsize = (20,7))

#이미지 다섯개를 무작위로 고른다.
radom_imgs = random.sample(range(output.shape[0]), 5) 

#원본(입력) 이미지를 맨위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[radom_imgs[i]].reshape(40,40), cmap = 'gray')
    if i==0:
        ax.set_ylabel('INPUT', SIZE = 20)
    ax.grid()
    ax.set_xticks([])    
    ax.set_yticks([])  

#잡음을 넣은 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[radom_imgs[i]].reshape(40,40), cmap = 'gray')
    if i==0:
        ax.set_ylabel('NOISE', SIZE = 20)
    ax.grid()
    ax.set_xticks([])    
    ax.set_yticks([])   

#출력 이미지
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[radom_imgs[i]].reshape(40,40), cmap = 'gray')
    if i==0:
        ax.set_ylabel('OUTPUT', SIZE = 20)
    ax.grid()
    ax.set_xticks([])    
    ax.set_yticks([]) 

plt.tight_layout()
plt.show() 
    
    

''' # 2. 모델
def build_model(drop=0.5, optimizer=Adam, filters=100, kernel_size=2, learning_rate=0.1):
    model = Sequential()
    model.add(Conv2D(16,(3,3),activation='relu', input_shape=(40, 40, 3),padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Conv2D(40,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(40,(5,5),activation='relu',padding='same')) 
    model.add(BatchNormalization())
    model.add(Conv2D(40,(5,5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(40,(5,5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3,3)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(40,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(40,(5,5),activation='relu',padding='same')) 
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3,3)))
    model.add(Dropout(0.3))
    
    model.add(Flatten())

    model.add(Dense(140,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(40,activation='relu'))
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
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),batch_size=8, epochs=400, callbacks = [er,lr])

loss, acc = model.evaluate(x_test, y_test, batch_size=8)
print('loss, acc : ', loss, acc) '''