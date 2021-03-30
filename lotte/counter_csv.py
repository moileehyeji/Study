import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from collections import Counter

num = 21
x = []
for i in range(num):           # 파일의 갯수
    df = pd.read_csv(f'C:/data/lotte/counter_csv/answer{i}.csv', index_col=0, header=0)
    data = df.to_numpy()
    x.append(data)

x = np.array(x)

# print(x.shape)
a1= []
a2= []
a3= []
df = pd.read_csv(f'C:/data/lotte/counter_csv/answer{i}.csv', index_col=0, header=0)
for i in range(72000):
    for j in range(1):
        b = []
        for k in range(num):         # 파일의 갯수
            b.append(x[k,i,j].astype('int'))
        # ======================================
        max1 = []
        max2 = []
        max3 = []

        count = Counter(b)

        # print(max(count,key=count.get))   #1개
        max_list = [k for k,v in count.items() if max(count.values()) == v] # 여러개


        max1 = max_list[0]
        a1.append(max1)

        # 세번째 어펜드
        if(len(max_list) > 2 ):
            max3 = max_list[2]
            a3.append(max3)
        else:
            max1 = max_list[0]
            a3.append(max1)

        # 두번째 어펜드
        if(len(max_list) == 2 ):
            max2 = max_list[1]
            a2.append(max2)
        else:
            max1 = max_list[0]
            a2.append(max1)
        

a1 = np.array(a1)
a2 = np.array(a2)
a3 = np.array(a3)
print(a1.shape, a2.shape, a3.shape) #(72000,) (72000,) (72000,)


sub = pd.read_csv('C:/data/lotte/csv/sample.csv')
sub['prediction'] = a1
sub.to_csv(f'C:/data/lotte/counter_csv/answer_add/final1.csv',index=False)
sub['prediction'] = a2
sub.to_csv(f'C:/data/lotte/counter_csv/answer_add/final2.csv',index=False)
sub['prediction'] = a3
sub.to_csv(f'C:/data/lotte/counter_csv/answer_add/final3.csv',index=False)



#-----------------
# answer_add2_5  :81.096 (0-4)
# answer2_add3_5 :80.635 (5-9)
# answer3_add1_5 :80.061 (10-14)
# answer4_add1_5 :78.419 (15-19)
# answer5_add2_5 :81.560 (0,1,5,10,15)
# answer6_add1_8 :81.467 (1,5,7,11,13,17,18,19)
# answer7_add2_6 :80.967 (2,4,6,8,11,16)
# answer8_add3_6 :81.072 (3,4,6,10,13,17)

#-----------------answer_all2_8 :82.504


# ============================
# answer_all2_8 :82.504
# answer9_add1_8 : 82.414 (6,7,9,11,13,16,17,19)
# answer10_add1_7: 81.786 (8,11,13,14,15,16,17)
# answer11_add1_8: 82.290 (2,3,5,7,10,12,13,17)
# answer12_add3_7: 82.146 (8,9,10,11,13,16,19)
# answer13_add1_7: 82.303 (7,8,9,10,11,16,17)
# answer14_add1_7: 82.553 (3,5,7,10,13,15,16)
# answer15_add1_7: 82.011 (6,7,8,13,14,16,17)
# ===========================answer2_all2_8 :83.265



# answer16_add1_7 :80.146 (7,10,13,14,17,18,19)
# answer17_add1_8 :82.444 82.215 82.310(6,7,9,11,13,15,17,19)