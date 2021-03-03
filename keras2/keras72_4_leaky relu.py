import numpy as np
import matplotlib.pyplot as plt

def leaky_relu(x, alpha = 0.01):
    return np.maximum(alpha*x,x)
# relu는 0보다 작은 입력신호에 대해 출력을 꺼버린다.
# 이로인해 발생할 수 있는 데이터 손실을 해결하기 위해 0보다 작은경우,
# 0에 근접하는 매우 작은 값으로 변환되도록한다.
# 그러나 relu에 비해 연산의 복잡성이 크다

x = np.arange(-5,5)
y = leaky_relu(x)

print(x)
print(y)

plt.plot(x,y)
plt.grid()
plt.show()


###과제
# elu, selu, leaky relu
# 72_2,3,4번으로 파일 만들기