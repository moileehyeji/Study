import numpy as np
import matplotlib.pyplot as plt

# alpha와 scale은 미리 정해지는 상수
# alpha와 scale의 값은 입력 입력의 평균값과 분산 값이 두 개의 연속되는 layer-사이에서 보존되도록 결정
def elu(x, alp):
    return (x>0)*x + (x<=0)*(alp * (np.exp(x)-1))

def selu(x, scale = 1.0507, alpha = 1.6733):
    return scale * elu(x, alpha)
# relu와 비슷하지만 음수 값에서 올라올 때 부드럽게 만들어 준다.
# gradient가 음수의 값이면 무조건 0으로 변환되면서 데이터손실의 가능성이 있는 relu보완

x = np.arange(-5,5,0.1)
y = selu(x)

print(x)
print(y)

plt.plot(x,y)
plt.grid()
plt.show()


###과제
# elu, selu, leaky relu
# 72_2,3,4번으로 파일 만들기