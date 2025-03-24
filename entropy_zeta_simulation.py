import numpy as np
import matplotlib.pyplot as plt
from scipy.special import zeta
from scipy.fft import fft, fftfreq

def zeta_entropy_signal(n_points=1024, beta=1.5):
    x = np.linspace(1, n_points, n_points)
    weights = zeta(beta, x)
    signal = np.sin(2 * np.pi * x / 64) * weights
    return x, signal, weights

x, signal, weights = zeta_entropy_signal()

plt.figure(figsize=(10, 4))
plt.plot(x, signal, label='Zeta-weighted Signal')
plt.title('Zeta-Weighted Entropy Signal')
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.savefig('results/zeta_entropy_signal.png')
plt.show()