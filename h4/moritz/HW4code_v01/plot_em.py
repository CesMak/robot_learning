import numpy as np
import pickle
import matplotlib.pyplot as plt

numDim = 10
numSamples = 25
maxIter = 100
numTrials = 10

lam_c = np.array([25, 7, 3])
Mu_w_c, Sigma_w_c, R_avg_c = pickle.load(open('save.p', 'rb'))

R_avg_mean = np.mean(R_avg_c, axis=1)
R_avg_std = np.std(R_avg_c, axis=1)

colors = ['r', 'b', 'g']
fig1 = plt.figure(4)
ax1 = fig1.add_subplot(1,1,1)
for i in xrange(lam_c.shape[0]):
    for j in xrange(numTrials):
        ax1.semilogy(np.abs(R_avg_c[i, j, :]), color=colors[i], alpha=0.2)
    ax1.semilogy(np.abs(R_avg_mean[i, :]), label=str(lam_c[i]), color=colors[i])
    ax1.semilogy(np.abs(R_avg_mean[i, :]) + 2 * R_avg_std[i, :], alpha=0.5, color=colors[i])
    ax1.semilogy(np.abs(R_avg_mean[i, :]) - 2 * R_avg_std[i, :], alpha=0.5, color=colors[i])

plt.legend()
plt.show()
