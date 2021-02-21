# 예측값으로 그래프 그리기      > 무진님꺼 따라하기 

# from google.colab import drive
# drive.mount('/content/drive')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from pandas import DataFrame


pred = pd.read_csv('./dacon/submission/unite_submission_12.csv')
# pred = pd.read_csv('./dacon/submission/unite_submission_11.csv')


# pred = pd.read_csv('./dacon/submission/unite_submission_10_Conv1D.csv')
# pred = pd.read_csv('./dacon/submission/unite_submission_10_Dense.csv')


# pred = pd.read_csv('./dacon/submission/unite_submission_add_column4.csv')
# pred = pd.read_csv('./dacon/submission/unite_submission_add_column3.csv')
# pred = pd.read_csv('./dacon/submission/unite_submission_add_column2.csv')
# pred = pd.read_csv('./dacon/submission/unite_submission_add_column.csv')


# pred = pd.read_csv('./dacon/submission/[submit0125(2)]unite_submission_addmodel.csv')
# pred = pd.read_csv('./dacon/submission/[submit0125(3)]unite_submission_td.csv')
# pred = pd.read_csv('./dacon/submission/[sumit]0.26(1)unite_submission_td_addmodel2.csv')

# pred = pd.read_csv('./dacon/submission/unite_submission_30minute_6_GHI.csv')



ranges = 672        # 7일치로 보겠음
hours = range(ranges)
pred = pred[ranges:ranges+ranges]

q_01 = pred['q_0.1'].values
q_02 = pred['q_0.2'].values
q_03 = pred['q_0.3'].values
q_04 = pred['q_0.4'].values
q_05 = pred['q_0.5'].values
q_06 = pred['q_0.6'].values
q_07 = pred['q_0.7'].values
q_08 = pred['q_0.8'].values
q_09 = pred['q_0.9'].values

import matplotlib.pyplot as plt

plt.figure(figsize=(18,2.5))
plt.subplot(1,1,1)
plt.plot(hours, q_01, color='red')
plt.plot(hours, q_02, color='#aa00cc')
plt.plot(hours, q_03, color='#00ccaa')
plt.plot(hours, q_04, color='#ccaa00')
plt.plot(hours, q_05, color='#00aacc')
plt.plot(hours, q_06, color='#aacc00')
plt.plot(hours, q_07, color='#cc00aa')
plt.plot(hours, q_08, color='#000000')
plt.plot(hours, q_09, color='blue')
# plt.title('[submit0125(2)]unite_submission_addmodel(1.9498123625)')
# plt.title('unite_submission_add_column')
# plt.title('unite_submission_add_column2')
# plt.title('unite_submission_add_column3')
# plt.title('unite_submission_add_column4')
# plt.title('unite_submission_30minute_6_GHI(3.4142240883)')
# plt.title('unite_submission_10_Dense')
# plt.title('unite_submission_10_Conv1D')
# plt.title('unite_submission_11')
plt.title('unite_submission_12')



plt.grid()
plt.legend()
plt.show()


'''
#======================================================================loss 시각화 
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))      

for i in range(9):
    plt.subplot(3,3,(i+1))  #2행 1열중 첫번째
    plt.plot(hist_list[i].history['loss'], marker='.', c='red', label='loss')
    plt.plot(hist_list[i].history['val_loss'], marker='.', c='blue', label='val_loss')
    plt.grid()

    # plt.title('손실비용')
    plt.title('Cost Loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(loc='upper right')

plt.show()
#=========================================================================
'''