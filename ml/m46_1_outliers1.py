# 이상치 처리방법 111111111111111111111111
# 이상치 자리를 nan으로 바꾼후 보간법
# 1. 0처리
# 2. Nan 처리 후 보간
# 3. 3,4,5...알아서 해

# 0사분위수(Q0): 최소값
# 1사분위수(Q1): 최소값 ~ 25% 번째 값
# 2사분위수(Q2): 중앙값
# 3사분위수(Q3): 중앙값 ~ 75% 번째 값
# 4사분위수(Q4): 최대값

import numpy as np
aaa = np.array([1,2,3,4,6,7,90,100,5000,10000])

#컬럼이 하나일때
def outliers (data_out):
    quartile_1, quartile_2, quartile_3 = np.percentile(data_out, [25,50,75])    #percentile:지정된 축을 따라 데이터의 q 번째 백분위 수를 계산합니다.
    print('1사분위 : ', quartile_1)
    print('2사분위 : ', quartile_2)
    print('3사분위 : ', quartile_3)
    iqr = quartile_3 - quartile_1   # 3사분위 - 1사분위(사분위수 범위)
    # 양방향으로 1.5배씩 늘려서 정상적인 데이터범위 지정
    # 통상 1.5
    lower_bound = quartile_1 - (iqr * 1.5)  
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

outlier_loc = outliers(aaa)
print('이상치의 위치 : ', outlier_loc)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()