import numpy as np
import matplotlib.pyplot as plt

def funcao(x):
    return x*np.sin(10*np.pi * x) + 1

x = np.linspace(-1, 2, 0.01)
y = funcao(x)
plt.plot(x, y)
plt.show()