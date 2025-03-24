import numpy as np
import matplotlib.pyplot as plt

def mcmc_entropy_convergence(steps=1000):
    entropy_vals = []
    current = 0.5
    for step in range(steps):
        proposal = current + np.random.normal(0, 0.05)
        accept_prob = min(1, np.exp(-(proposal**2 - current**2)))
        if np.random.rand() < accept_prob:
            current = proposal
        entropy_vals.append(-current * np.log(np.abs(current) + 1e-10))
    return entropy_vals

entropy_vals = mcmc_entropy_convergence()

plt.figure(figsize=(10, 4))
plt.plot(entropy_vals)
plt.title('Bayesian MCMC Entropy Convergence')
plt.xlabel('Steps')
plt.ylabel('Entropy')
plt.grid(True)
plt.savefig('results/mcmc_entropy_convergence.png')
plt.show()