import numpy as np

# split 함수만들기 (다:1)
'''
import numpy as np

data = np.array([1,2,3,4,5,6,7,8,9,10])

def split_xy1 (data, time_steps):
    x, y = [], []
    for i in range(len(data)):
        end_number = i + time_steps
        if end_number > len(data) - 1:
            break
        tmp_x, tmp_y = data[i:end_number], data[end_number]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy1(data, 4)
print(x, '\n', y)

'''

# split 함수만들기 (다:다)

data = np.array([1,2,3,4,5,6,7,8,9,10])
time_steps = 4
y_col = 2

def split_xy2(data, time_steps, y_col):
    x, y = [],[]

    for i in range(len(data)):

        x_end_number = i + time_steps
        y_end_number = x_end_number + y_col

        if y_end_number > len(data):
            break

        tmp_x = data[i : x_end_number]
        tmp_y = data[x_end_number : y_end_number]

        x.append(tmp_x) 
        y.append(tmp_y)

    return np.array(x), np.array(y)

x, y = split_xy2(data, time_steps, y_col)
print(x, '\n', y)
print(x.shape)
print(y.shape)

    