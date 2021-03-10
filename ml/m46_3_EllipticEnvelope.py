# 이상치 처리방법 222222222222222222222
# EllipticEnvelope
# 가우스 분산 데이터 세트에서 이상 값을 감지하기위한 개체

from sklearn.covariance import EllipticEnvelope
import numpy as np

aaa = np.array([[1,2,-1000,3,4,6,7,8,90,100,5000],
                [1000,2000,3,4000,5000,6000,7000,8,9000,10000,1001]])
aaa = np.transpose(aaa)
print(aaa.shape)    #(11, 1)


#--------------------------------------------EllipticEnvelope
# 가우스 분산 데이터 세트에서 이상 값을 감지하기위한 개체
outlier = EllipticEnvelope(contamination=.2)    #데이터 세트의 오염 정도, 즉 데이터 세트의 이상 값 비율. 범위는 (0, 0.5)
outlier.fit(aaa)

# OUTLIERS 예측
pred = outlier.predict(aaa) 
print(pred) #[ 1  1 -1  1  1  1  1  1  1  1 -1]
# 예측값 한줄 WHY?
# [1,   2,    -1000, 3,    4,   6,   7,   8, 90,   100,    5000]
# [1000,2000, 3,     4000, 5000,6000,7000,8, 9000, 10000,  1001]
# 2 컬럼을 한개로 보고 예측


#----------------------------------------
# aaa = np.array([[1,2,-1000,3,4,6,7,8,90,100,5000],          
# ---> standard scalar 전처리하면 데이터가 모여있지않고 흩어질 것
# 그렇다면 중위값을 1로 scaling하자
# m46_4_rubust.py