import numpy as np
import matplotlib.pyplot as plt

def triangle_wave(x: float):
    # Adjust the triangle wave to have the correct orientation
    if x > 0:
        return (1 - abs((x % 2) - 1))
    else:
        return - (1 - abs((x % 2) - 1))

x = np.linspace(-3, 3, 1000)
y = [triangle_wave(i) for i in x]

plt.plot(x, y)
plt.xlabel('Input')
plt.ylabel('Triangle Wave')
plt.title('Triangle Wave')
plt.grid(True)
plt.show()
