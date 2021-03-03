import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0,x)
# 매우 간결한 함수이다. 0보다 크면x,작으면0이다.
# 연산이 간결해서 학습속도가 빠르다.
# 그러나 음수의 값이면 무조건 0으로 변환되면서 데이터손실의 가능성이 있다.

x = np.arange(-5,5,0.1)
y = relu(x)

print(x)
print(y)

plt.plot(x,y)
plt.grid()
plt.show()


###과제
# elu, selu, reaky relu
# 72_2,3,4번으로 파일 만들기