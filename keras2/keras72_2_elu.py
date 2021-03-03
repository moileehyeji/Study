import numpy as np
import matplotlib.pyplot as plt

def elu(x, alpha = 1.6733):
    return (x>0)*x + (x<=0)*(alpha * (np.exp(x)-1))
# alpha와 scale은 미리 정해지는 상수
# alpha와 scale의 값은 입력 입력의 평균값과 분산 값이 두 개의 연속되는 layer-사이에서 보존되도록 결정
#개형은 Relu와 유사하며 0ㅂㅎ다 작은 경우는 alpha값을 이용해서 그래프를 부드럽게 만든다.
#때문에, elu를 미분해도 부드럽게 이어지는 모습을 확인할 수 있다.

x = np.arange(-5,5,0.1)
y = elu(x)

print(x)
print(y)

plt.plot(x,y)
plt.grid()
plt.show()


###과제
# elu, selu, leaky relu
# 72_2,3,4번으로 파일 만들기