import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Dropout,Input, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from keras.optimizers import RMSprop

(x_train, y_train),(x_test, y_test) = mnist.load_data()

# ========= 1. 전처리 ==================================
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# one-hot
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# =============== 2. 모델링 =======================
input_dim = x_train.shape[1] # 784
num_classes = 10

def my_model( init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, kernel_initializer=init, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, kernel_initializer=init, activation='relu'))
    model.add(Dense(num_classes, kernel_initializer=init, activation='softmax'))

    # 훈련
    model.compile(loss='categorical_crossentropy', 
                 optimizer=RMSprop(),
                  metrics=['accuracy'])
    return model

model2 = KerasClassifier(build_fn=my_model, verbose=1)

init_mode = ['glorot_uniform', 'uniform'] 
batches = [128, 512]
epochs = [10, 20]

param_grid = dict(epochs=epochs, batch_size=batches, init=init_mode)

grid = RandomizedSearchCV(model2, 
                    param_grid,
                    cv=3)
grid_result = grid.fit(x_train, y_train)

# print results
print(f'Best Accuracy for {grid_result.best_score_:.4} using {grid_result.best_params_}')
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f'mean={mean:.4}, std={stdev:.4} using {param}')