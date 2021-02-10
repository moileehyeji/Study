
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
    validation_split=0.15
)
test_dategen = ImageDataGenerator(rescale=1./255) # test 이미지는 증폭XXX, 0-1사이값으로 정규화

train_data = 1476
val_data = 260
xy_train = train_datagen.flow_from_directory(
    '../data/Image/gender',
    target_size = (56,56),                
    batch_size = train_data,
    class_mode = 'binary',               
    subset = 'training'
)
xy_val = train_datagen.flow_from_directory(
    '../data/Image/gender',
    target_size = (56,56),               
    batch_size = val_data,
    class_mode = 'binary',               
    subset = 'validation'
)
# Found 1216 images belonging to 3 classes.
# Found 520 images belonging to 3 classes.


# train, test npy 저장
np.save('../data/Image/gender/npy/keras67_4_train_x.npy', arr=xy_train[0][0])
np.save('../data/Image/gender/npy/keras67_4_train_y.npy', arr=xy_train[0][1])
np.save('../data/Image/gender/npy/keras67_4_val_x.npy', arr=xy_val[0][0])
np.save('../data/Image/gender/npy/keras67_4_val_y.npy', arr=xy_val[0][1])
