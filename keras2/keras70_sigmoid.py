import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1+ np.exp(-x))  #0,1사이 수렴
# 무조건 0과1사이의 값으로 반환한다. 
# gradient에 기반한 학습을 하는 backpropagation방식으로 학습을 진행할때 층을 거듭할수록
# 값이 매우 작아지는 문제가 발생한다. 
# 0과 1사이의 값을 계속 곱한다면 0에 수렴하는 것과 같은 논리이다.
# 흔히 vanishing gradient문제라고도 한다. 
# np.exp : 자연상수 e의 제곱값을 반환
# np.exp(-x): e^-x

x = np.arange(-5, 5, 0.1)
y = sigmoid(x)

print(x)
print(y)

plt.plot(x,y)
plt.grid()
plt.show()