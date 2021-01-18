import numpy as np

# split 함수만들기 (다 입력, 다:1)
'''
import numpy as np

data = np.array([[1,2,3,4,5,6,7,8,9,10],
                 [11,12,13,14,15,16,17,18,19,20],
                 [21,22,23,24,25,26,27,28,29,30]])

data = np.transpose(data)
print(data)
print(data.shape)   #(10, 3)

def split_xy3 (data, time_steps, y_col):
    
    x, y = [], []

    for i in range(len(data)):

        x_end_number = i + time_steps
        y_end_number = x_end_number + y_col - 1

        # if end_number > len(data) - 1:
        #     break
        if y_end_number > len(data):
            break

        tmp_x = data[i:x_end_number, :-1]
        tmp_y = data[x_end_number - 1 : y_end_number, -1]

        x.append(tmp_x)
        y.append(tmp_y)

    return np.array(x), np.array(y)

x, y = split_xy3(data, 3, 1)

print(x, '\n', y)
print(x.shape)
print(y.shape)
'''

# split 함수만들기 (다 입력, 다:다)
'''
import numpy as np
data = np.array([[1,2,3,4,5,6,7,8,9,10],
                 [11,12,13,14,15,16,17,18,19,20],
                 [21,22,23,24,25,26,27,28,29,30]])

data = np.transpose(data)
print(data)
print(data.shape)   #(10, 3)

def split_xy4 (data, time_steps, y_col):
    
    x, y = [], []

    for i in range(len(data)):

        x_end_number = i + time_steps
        y_end_number = x_end_number + y_col - 1

        # if end_number > len(data) - 1:
        #     break
        if y_end_number > len(data):
            break

        tmp_x = data[i:x_end_number, :-1]
        tmp_y = data[x_end_number - 1 : y_end_number, -1]

        x.append(tmp_x)
        y.append(tmp_y)

    return np.array(x), np.array(y)

x, y = split_xy4(data, 3, 2)

print(x, '\n', y)
print(x.shape)  #(7, 3, 2)
print(y.shape)  #(7, 2)
'''

# split 함수만들기 (다 입력, 다:다 두번째)
'''
import numpy as np
data = np.array([[1,2,3,4,5,6,7,8,9,10],
                 [11,12,13,14,15,16,17,18,19,20],
                 [21,22,23,24,25,26,27,28,29,30]])

data = np.transpose(data)
print(data)
print(data.shape)   #(10, 3)

def split_xy5 (data, time_steps, y_col):
    
    x, y = [], []

    for i in range(len(data)):

        x_end_number = i + time_steps
        y_end_number = x_end_number + y_col 

        # if end_number > len(data) - 1:
        #     break
        if y_end_number > len(data):
            break

        tmp_x = data[i:x_end_number, :]
        tmp_y = data[x_end_number : y_end_number, :]

        x.append(tmp_x)
        y.append(tmp_y)

    return np.array(x), np.array(y)

x, y = split_xy5(data, 3, 1)

print(x, '\n', y)
print(x.shape)  #(7, 3, 3)
print(y.shape)  #(7, 1, 3)
'''

# split 함수만들기 (다 입력, 다:다 세번째)

import numpy as np
data = np.array([[1,2,3,4,5,6,7,8,9,10],
                 [11,12,13,14,15,16,17,18,19,20],
                 [21,22,23,24,25,26,27,28,29,30]])

data = np.transpose(data)
print(data)
print(data.shape)   #(10, 3)

def split_xy4 (data, x_col, x_row, y_col, y_row):
    
    x, y = [], []

    for i in range(len(data)):

        x_end_number = i + x_row
        y_end_number = x_end_number + y_row 

        if y_end_number > len(data):
            break

        tmp_x = data[i : x_end_number, :-1]
        tmp_y = data[x_end_number - 1 : y_end_number, -1]

        x.append(tmp_x)
        y.append(tmp_y)

    return np.array(x), np.array(y)

x, y = split_xy4(data, 2, 3, 2, 1)

print(x, '\n', y)
print(x.shape)  #(7, 3, 2)
print(y.shape)  #(7, 2)
