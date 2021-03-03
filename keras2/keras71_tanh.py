import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)
y = np.tanh(x)
# sigmoid의 vanishing gradient문제는 여전히 남아있다. 
# 그러나 중심을 원점으로 옮김으로써 학습과정이 sigmoid에 비해 최적화되었다고 할 수 있다.

plt.plot(x,y)
plt.grid()
plt.show()