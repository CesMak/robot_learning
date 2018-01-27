import numpy as np
import pickle
import matplotlib.pyplot as plt
from Pend2dBallThrowDMP import *

numDim = 10
numSamples = 25
maxIter = 100
numTrials = 3

env = Pend2dBallThrowDMP()

learning_rate = np.array([0.1])#, 0.2, 0.4])

Mu_w_c, Sigma_w_c, R_avg_c = pickle.load(open('save_pg_b.p', 'rb'))

R_avg_mean = np.abs(np.mean(R_avg_c, axis=1))
R_avg_std = np.std(R_avg_c, axis=1)

colors = ['r', 'b', 'g']
fig1 = plt.figure(4)


for i in xrange(learning_rate.shape[0]):
    #plt.subplot(learning_rate.shape[0], 1, i+1)

    #for j in xrange(numTrials):
    #    plt.plot(np.abs(R_avg_c[i, j, :]), color=colors[i], alpha=0.4)
    plt.plot(R_avg_mean[i, :], label=str(learning_rate[i]), color=colors[i])
    plt.plot(R_avg_mean[i, :] + 2 * R_avg_std[i, :], alpha=0.5, color=colors[i])
    plt.plot(R_avg_mean[i, :] - 2 * R_avg_std[i, :], alpha=0.5, color=colors[i])
    plt.fill_between(np.arange(0, 100, 1), R_avg_mean[i, :] + 2 * R_avg_std[i, :], R_avg_mean[i, :] - 2 * R_avg_std[i, :],
                     alpha=0.5, color=colors[i])
    #plt.ylim((1e1, 2e7))
    plt.yscale("log", nonposy="clip")
    plt.legend()

plt.legend()
plt.show()
