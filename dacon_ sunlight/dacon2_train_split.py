import numpy as np

# ================================shift 전=======================================================
data = np.load('./dacon/npy/dacon_train.npy')

print(data.shape)   #(52560, 7)

def split_xy4 (data, time_steps, y_col):
    
    x, y = [], []

    for i in range(len(data)):

        x_end_number = i + time_steps
        y_end_number = x_end_number + y_col 

        if y_end_number > len(data):
            break

        tmp_x = data[i:x_end_number, :]
        tmp_y = data[x_end_number : y_end_number, -1]

        x.append(tmp_x)
        y.append(tmp_y)

    return np.array(x), np.array(y)

x, y = split_xy4(data, 7*24*2, 24*2*2)

print('x : ', x)    
print('y : ', y)
print(x.shape)      #(52129, 336, 7)
print(y.shape)      #(52129, 96)
print(y[16:30,:])

np.save('./dacon/npy/dacon_train_x.npy', arr=x)
np.save('./dacon/npy/dacon_train_y.npy', arr=y)

# predict 값 to_csv
import pandas as pd
# df_x = pd.DataFrame(x)
df_y = pd.DataFrame(y)

# df_x.to_csv('./dacon/csv/dacon_train_x.csv')
df_y.to_csv('./dacon/csv/dacon_train_y.csv')

# ===============================================================================================

'''
# ================================shift 후=======================================================

data = np.load('./dacon/npy/dacon_train_shift.npy')

x = data[:,:-2]
y = data[:,7:]

print('x : ', x)    
print('y : ', y)
print(x.shape)     # (52560, 7)
print(y.shape)     # (52560, 2)

np.save('./dacon/npy/dacon_train_shift_x.npy', arr=x)
np.save('./dacon/npy/dacon_train_shift_y.npy', arr=y)

'''