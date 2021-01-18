import numpy as np
from sklearn.datasets import load_iris

dataset = load_iris()
print(dataset)
'''
출력 :  딕셔너리 형태(Key : Value)로 저장되어있다.  -->dataset['data']...
{'data': array([[5.1, 3.5, 1.4, 0.2],
       [4.9, 3. , 1.4, 0.2],  ...
'''

# print(dataset.keys())   #dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

x_data = dataset.data       #dictionary Key
y_data = dataset.target     #dictionary Key

'''
Dictionary 용법=========
x_data = dataset['data']
y_data = dataset['target']
print(x_data)
print(y_data)
=========================
'''

print(dataset.frame)            #None
print(dataset.target_names)     #['setosa' 'versicolor' 'virginica']
print(dataset['DESCR'])
print(dataset['feature_names'])
print(dataset.filename)

# x, y Numpy저장
print(type(x_data), type(y_data))         #<class 'numpy.ndarray'> <class 'numpy.ndarray'>

np.save('../data/npy/iris_x.npy', arr=x_data)
np.save('../data/npy/iris_y.npy', arr=y_data)


