# x1 : x_predict --> y_predict 검증용   (shape : (80, 96))
# x2 : 제출용                           (shape : (81, 96))

import numpy as np

# ================================shift 전=======================================================
data = np.load('./dacon/npy/dacon_test.npy')

print(data.shape)   #(27216, 6)

#==================================# x1 : x_predict --> y_predict 검증용==========================
def split_xy4 (data, time_steps, y_col):
    
    x, y = [], []

    for i in range(len(data)):

        x_start_number = i * time_steps 
        x_end_number = x_start_number + time_steps - 1
        y_start_number = x_end_number + 1 
        y_end_number = x_end_number + y_col 

        if y_end_number > len(data):
            break

        tmp_x = data[x_start_number : x_end_number + 1, :]
        tmp_y = data[y_start_number : y_end_number + 1, -1]

        x.append(tmp_x)
        y.append(tmp_y)

    return np.array(x), np.array(y)

x1, y1 = split_xy4(data, 7*24*2, 24*2*2)


print('x : ', x1)    
print('y : ', y1)
print(x1.shape)      #(80, 336, 7)
print(y1.shape)      #(80, 96)
print(y1[:1,:])

np.save('./dacon/npy/dacon_test_x1.npy', arr=x1)
np.save('./dacon/npy/dacon_test_y1.npy', arr=y1)


# predict 값 to_csv
import pandas as pd
# df_x = pd.DataFrame(x)
df_y = pd.DataFrame(y1)

# df_x.to_csv('./dacon/csv/dacon_train_x.csv')
df_y.to_csv('./dacon/csv/dacon_test_y1.csv')

#==============================================================================================

'''

# ================================shift 후======================================================

data = np.load('./dacon/npy/dacon_test_shift.npy')

x = data[:,:-2]
y = data[:,7:]

print('x : ', x)    
print('y : ', y)
print(x.shape)     # (52560, 7)
print(y.shape)     # (52560, 2)

np.save('./dacon/npy/dacon_train_shift_x.npy', arr=x)
np.save('./dacon/npy/dacon_train_shift_y.npy', arr=y)

'''