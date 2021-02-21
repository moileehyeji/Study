# x1 : x_predict --> y_predict 검증용
# x2 : 제출용

import numpy as np

data = np.load('./dacon/npy/dacon_test.npy')

print(data.shape)   #(27216, 6)

  
#==================================# x2 제출용==================================================

def split_xy4 (data, time_steps):
    
    x = []

    for i in range(len(data)):

        x_start_number = i * time_steps 
        x_end_number = x_start_number + time_steps - 1

        if x_end_number > len(data):
            break

        tmp_x = data[x_start_number : x_end_number + 1, :]

        x.append(tmp_x)

    return np.array(x)

x2 = split_xy4(data, 7*24*2)

print('x : ', x2)   
print(x2.shape)     # (81, 336, 7) 

np.save('./dacon/npy/dacon_test_submit_x2.npy', arr=x2)

#==============================================================================================