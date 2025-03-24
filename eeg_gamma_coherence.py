import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence

def generate_eeg_like_signal(freq=40, length=5, fs=250, phase_shift=0.1):
    t = np.linspace(0, length, int(fs*length), endpoint=False)
    base = np.sin(2 * np.pi * freq * t)
    harmonic = np.sin(2 * np.pi * freq * t + phase_shift)
    noise = 0.3 * np.random.randn(len(t))
    return base + noise, harmonic + noise, t

sig1, sig2, t = generate_eeg_like_signal()

f, Cxy = coherence(sig1, sig2, fs=250)

plt.figure(figsize=(10, 4))
plt.semilogy(f, Cxy)
plt.title('EEG-like Gamma Coherence (40 Hz)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Coherence')
plt.grid(True)
plt.savefig('results/eeg_gamma_coherence.png')
plt.show()