import numpy as np
import pickle
import matplotlib.pyplot as plt
from Pend2dBallThrowDMP import *

numDim = 10
numSamples = 25
maxIter = 100
numTrials = 10

env = Pend2dBallThrowDMP()
lam_c = np.array([25, 7, 3])
Mu_w_c, Sigma_w_c, R_avg_c = pickle.load(open('save_pg_01_with_plus.p', 'rb'))

R_avg_mean = np.mean(R_avg_c, axis=0)
R_avg_std = np.std(R_avg_c, axis=0)

colors = ['r', 'b', 'g']
fig1 = plt.figure(4)
ax1 = fig1.add_subplot(1,1,1)
plt.yscale("log", nonposy="clip")

for i in xrange(numTrials):
    plt.plot(np.abs(R_avg_c[i, :]), color=colors[1], alpha=0.2)
    #env.animate_fig(np.random.multivariate_normal(Mu_w_c[i,:], Sigma_w_c[i,:,:]))
plt.plot(R_avg_mean, color=colors[1], label='alpha=0.1')
plt.plot(R_avg_mean + 2 * R_avg_std, alpha=0.6, color=colors[1])
plt.plot(R_avg_mean - 2 * R_avg_std, alpha=0.6, color=colors[1])
plt.fill_between(np.arange(0, 100, 1), R_avg_mean + 2 * R_avg_std, R_avg_mean - 2 * R_avg_std,
                     alpha=0.1, color=colors[1])

plt.legend()
plt.show()
