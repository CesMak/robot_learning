from matplotlib import pyplot as plt
import numpy as np
from Pend2dBallThrowDMP import *
import pickle

np.seterr(all='raise')
env = Pend2dBallThrowDMP()

numDim = 10
numSamples = 25
maxIter = 100
numTrials = 10

# Do your learning


def PolicyGradient(alpha = 0.1):
    # For example, let initialize the distribution...
    Mu = np.zeros(numDim)
    Sigma = np.eye(numDim) * 10

    # Do your learning
    R_avg = np.zeros(maxIter)
    for i in xrange(maxIter):
        # reward
        R = np.zeros((numSamples, 1))
        theta = np.zeros((numSamples, numDim))
        for j in xrange(numSamples):
            theta[j, :] = np.random.multivariate_normal(Mu, Sigma)
            R[j] = env.getReward(theta[j, :])
        R_avg[i] = np.mean(R)
        #print 'R_mean_{}: {}'.format(i, R_avg)

        temp = np.zeros(theta.shape)
        for j in xrange(numSamples):
            temp[j, :] = np.dot(np.linalg.inv(Sigma), theta[j] + Mu) * (R[j]-R_avg[i])
                # gradient
        grad_mu = np.sum(temp, axis=0)

        Mu = Mu + alpha*grad_mu

        if i>1:
            if np.abs(R_avg[i] - R_avg[i-1]) < 1e-3:
                break
    print 'end'
    return Mu, Sigma, R_avg

Mu_w_b, Sigma_w_b, R_avg_b = PolicyGradient()
sample = np.random.multivariate_normal(Mu_w_b,Sigma_w_b)
reward = env.getReward(sample)
print 'R_mean_end: {}'.format(reward)

# Save animation
env.animate_fig ( np.random.multivariate_normal(Mu_w_b,Sigma_w_b) )

plt.savefig('EM-b.pdf')

plt.figure(3)
plt.plot(R_avg_b)
plt.title('2b) one run')

Mu_w_c = np.zeros((numTrials, numDim))
Sigma_w_c = np.zeros((numTrials, numDim, numDim))
R_avg_c = np.zeros((numTrials, maxIter))

for i in xrange(numTrials):
    Mu_w_c[i, :], Sigma_w_c[i, :, :], R_avg_c[i, :] = PolicyGradient()
pickle.dump([Mu_w_c, Sigma_w_c, R_avg_c], open('save_pg.p', 'wb'))

R_avg_mean_c = np.mean(R_avg_c, axis=0)
R_avg_std_c = np.std(R_avg_c, axis=0)

plt.figure(4)
plt.plot(R_avg_mean_c)
#plt.fill_between(R_avg_mean_c[0, :] + 2 * R_avg_std_c[0, :], R_avg_mean_c[0, :] - 2 * R_avg_std_c[0, :],
#                alpha=0.5)
plt.title('2b) 10 runs')

plt.show()
