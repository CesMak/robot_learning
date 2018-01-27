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
Mu_w_c, Sigma_w_c, R_avg_c = pickle.load(open('save_em.p', 'rb'))

R_avg_mean = np.mean(R_avg_c, axis=1)
R_avg_std = np.std(R_avg_c, axis=1)

colors = ['r', 'b', 'g']
fig1 = plt.figure(4)

for i in xrange(lam_c.shape[0]):
    plt.subplot(lam_c.shape[0], 1, i+1)
    plt.grid()
    plt.yscale("log", nonposy="clip")
    for j in xrange(numTrials):
        plt.plot(np.abs(R_avg_c[i, j, :]), color=colors[i], alpha=0.2)
    plt.plot(np.abs(R_avg_mean[i, :]), label=str(lam_c[i]), color=colors[i])
    plt.plot(np.abs(R_avg_mean[i, :]) + 2 * R_avg_std[i, :], alpha=0.5, color=colors[i])
    plt.plot(np.abs(R_avg_mean[i, :]) - 2 * R_avg_std[i, :], alpha=0.5, color=colors[i])

    plt.fill_between(np.arange(0, 100, 1), np.abs(R_avg_mean[i, :]) + 2 * R_avg_std[i, :], np.abs(R_avg_mean[i, :]) - 2 * R_avg_std[i, :],
                   alpha=0.5, color=colors[i])
    plt.ylim((1e1, 2e6))
    plt.xlim((0,100))
    plt.legend()

sample = np.random.multivariate_normal(Mu_w_c[1,0,:],Sigma_w_c[1,0,:,:])
reward = env.getReward(sample)
print 'R_mean_end: {}'.format(reward)

# Save animation
env.animate_fig ( sample )

plt.savefig('EM-b.pdf')

plt.show()
