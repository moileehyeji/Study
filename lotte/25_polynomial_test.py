import numpy as np
from tensorflow.keras.applications import EfficientNetB4, EfficientNetB2
from tensorflow.keras.applications.efficientnet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, GlobalMaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Dense, Activation, Dropout
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import VGG19, MobileNet, ResNet101
from tensorflow.keras.optimizers import Adam, SGD


#데이터 지정 및 전처리
x = np.load("C:/Study/lotte/data/npy/64_project_x.npy",allow_pickle=True)
x_pred = np.load('C:/Study/lotte/data/npy/64_test.npy',allow_pickle=True)
y = np.load("C:/Study/lotte/data/npy/64_project_y.npy",allow_pickle=True)

# print(x.shape, x_pred.shape, y.shape)   #(48000, 128, 128, 3) (72000, 128, 128, 3) (48000, 1000)

x = preprocess_input(x) # (48000, 255, 255, 3)
x_pred = preprocess_input(x_pred)   # 


''' 
idg = ImageDataGenerator(
    # rotation_range=10, acc 하락
    width_shift_range=(-1,1),  
    height_shift_range=(-1,1), 
    rotation_range=40, 
    # shear_range=0.2)    # 현상유지
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

idg2 = ImageDataGenerator() '''

x_train, x_valid, y_train, y_valid = train_test_split(x,y, train_size = 0.8, shuffle = True, random_state=66)
# print(x_train.shape, y_train.shape) #(38400, 128, 128, 3) (38400, 1000)
# print(x_valid.shape, y_valid.shape) #(9600, 128, 128, 3) (9600, 1000)


from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=True)
x_train = x_train.reshape(-1, 64*64*3)
poly_x_train = poly.fit_transform(x_train)
print('변환된 2차 다항식 계수 피처:\n', poly_x_train.shape)

# numpy.core._exceptions.MemoryError: Unable to allocate 21.1 TiB for an array with shape (38400, 75503617) and data type float64
# 크기64,128 다 터져서 test안됨