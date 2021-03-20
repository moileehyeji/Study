import numpy as np
import pandas as pd
from tensorflow.keras.applications.efficientnet import EfficientNetB0, EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

#0. 변수
batch = 256
seed = 42
dropout = 0.2
epochs = 10
model_path = 'C:/LPD_competition/lpd_001.hdf5'
sub = pd.read_csv('C:/LPD_competition/sample.csv', header = 0)
es = EarlyStopping(patience = 5)
lr = ReduceLROnPlateau(factor = 0.25, patience = 3, verbose = 1)
cp = ModelCheckpoint(model_path, save_best_only= True)

#1. 데이터
train_gen = ImageDataGenerator(
    validation_split = 0.2,
    width_shift_range=(-1,1), 
    height_shift_range=(-1,1),  
    shear_range=0.2,
    preprocessing_function= preprocess_input,
    rescale = 1/255.
)

test_gen = ImageDataGenerator(
    preprocessing_function= preprocess_input,
    rescale = 1/255.
)

# Found 39000 images belonging to 1000 classes.
train_data = train_gen.flow_from_directory(
    'C:/LPD_competition/train',
    target_size = (224, 224),
    class_mode = 'categorical',
    batch_size = batch,
    seed = seed,
    subset = 'training'
)

# Found 9000 images belonging to 1000 classes.
val_data = train_gen.flow_from_directory(
    'C:/LPD_competition/train',
    target_size = (224, 224),
    class_mode = 'categorical',
    batch_size = batch,
    seed = seed,
    subset = 'validation'
)

# Found 72000 images belonging to 1 classes.
test_data = test_gen.flow_from_directory(
    'C:/LPD_competition/t',
    target_size = (224, 224),
    class_mode = None,
    batch_size = batch,
    seed = seed,
    shuffle = False
)

#2. 모델
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation
from tensorflow.keras.applications import VGG19, MobileNet
# efficientnetb7 = EfficientNetB7(include_top=False,weights='imagenet',input_shape=(256, 256, 3))
# efficientnetb7.trainable = False
mobile_net = MobileNet(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
top_model = mobile_net.output
top_model = GlobalAveragePooling2D()(top_model)
top_model = Flatten()(top_model)
top_model = Dense(1024, activation="relu")(top_model)
top_model = Dense(1024, activation="relu")(top_model)
top_model = Dense(512, activation="relu")(top_model)
top_model = Dense(1000, activation="softmax")(top_model)

model = Model(inputs=mobile_net.input, outputs = top_model)


# mobile_net = MobileNet(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
# a = mobile_net.output
# a = GlobalAveragePooling2D() (a)
# a = Flatten() (a)
# a = Dense(128) (a)
# a = BatchNormalization() (a)
# a = Activation('relu') (a)
# a = Dense(64) (a)
# a = BatchNormalization() (a)
# a = Activation('relu') (a)
# a = Dense(1000, activation= 'softmax') (a)

# model = Model(inputs = mobile_net.input, outputs = a)
# eff = EfficientNetB7(include_top = False, input_shape = (256, 256, 3))
# eff.trainable = False

# model = Sequential()
# model.add(eff)
# model.add(MaxPooling2D(2, padding = 'same'))
# model.add(Conv2D(1280, 2, padding = 'same'))
# model.add(MaxPooling2D(2, padding = 'same'))
# model.add(BatchNormalization())
# model.add(Flatten())
# model.add(Dense(5120, activation = 'relu', kernel_initializer = 'he_normal'))
# model.add(Dropout(dropout))
# model.add(Dense(2560, activation = 'relu', kernel_initializer = 'he_normal'))
# model.add(Dense(1280, activation = 'relu', kernel_initializer = 'he_normal'))
# model.add(Dense(1000, activation = 'softmax'))

#3. 컴파일 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(train_data, steps_per_epoch = len(train_data), validation_data= val_data, validation_steps= len(val_data),\
    epochs = epochs, callbacks = [es, cp, lr])

model = load_model(model_path)

#4. 평가 예측
pred = model.predict(test_data)
pred = np.argmax(pred, 1)
sub.loc[:,'prediction'] = pred
sub.to_csv('C:/LPD_competition/sample_001.csv', index = False)