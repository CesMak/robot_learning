from matplotlib import pyplot as plt
import numpy as np
from Pend2dBallThrowDMP import *
import pickle
##matplotlib inline

env = Pend2dBallThrowDMP()

numDim = 10
numSamples = 25
maxIter = 100
numTrials = 10

def EM(lam = 7):
    # For example, let initialize the distribution...
    Mu_w = np.zeros(numDim)
    Sigma_w = np.eye(numDim) * 1e6

    R_mean_old = 0
    # Do your learning
    R_avg = np.zeros(maxIter)
    for i in xrange(maxIter):
        # reward
        R = np.zeros((numSamples, 1))
        theta = np.zeros((numSamples, numDim))
        for j in xrange(numSamples):
            theta[j, :] = np.random.multivariate_normal(Mu_w, Sigma_w)
            R[j] = env.getReward(theta[j, :])
        R_avg[i] = np.mean(R)
        #print 'R_mean_{}: {}'.format(i, R_avg)

        # weights
        beta = lam / (np.max(R) - np.min(R))
        w = np.exp((R - np.max(R)) * beta)

        Mu_w = np.sum(w * theta, axis=0) / np.sum(w)

        temp = np.zeros((numDim, numDim))

        # for k in xrange(numDim):
        #    diff = theta - Mu_w[k]

        for j in xrange(numSamples):
            diff = (theta[j, :] - Mu_w).reshape((numDim, 1))
            temp += w[j] * np.dot(diff, diff.T)
        Sigma_w = temp / np.sum(w) + np.eye(numDim)
        #    Sigma_w[j,:] = np.sum(w * np.dot((theta[j, :]-Mu_w[j]),(theta-Mu_w[j]).T), axis=0)/(np.sum(w)) + np.eye(numDim)

        # new mu
        # Mu_w = np.sum(w*theta, axis=0) / np.sum(w)
        # Sigma_w = (np.sum(w * np.dot((theta-Mu_w),(theta-Mu_w).T), axis=0))/(np.sum(w)) + np.eye(numDim)

        #if i>1:
        #    if np.abs(R_avg[i] - R_avg[i-1]) < 1e-3:
         #       break
    print 'end'
    return Mu_w, Sigma_w, R_avg



# ... then draw a sample and simulate an episode

# A4.1 b)
Mu_w_b, Sigma_w_b, R_avg_b = EM()
sample = np.random.multivariate_normal(Mu_w_b,Sigma_w_b)
reward = env.getReward(sample)
print 'R_mean_end: {}'.format(reward)

# Save animation
env.animate_fig ( np.random.multivariate_normal(Mu_w_b,Sigma_w_b) )

plt.savefig('EM-b.pdf')

plt.figure(3)
plt.plot(R_avg_b)


# A4.1 c)

lam_c = np.array([25, 7, 3])
Mu_w_c = np.zeros((lam_c.shape[0], numTrials, numDim))
Sigma_w_c = np.zeros((lam_c.shape[0], numTrials, numDim, numDim))
R_avg_c = np.zeros((lam_c.shape[0], numTrials, maxIter))

for i in xrange(lam_c.shape[0]):
    for j in xrange(numTrials):
        Mu_w_c[i, j, :], Sigma_w_c[i, j, :, :], R_avg_c[i, j, :] = EM(lam_c[i])

pickle.dump([Mu_w_c, Sigma_w_c, R_avg_c], open('save_em.p', 'wb'))
#R_avg_mean = np.zeros((lam_c.shape[0], maxIter))
#for i in xrange(lam_c.shape[0]):
#    R_avg_mean[i, :] = np.mean(R_avg_c[i, :, :], axis=1)
R_avg_mean = np.mean(R_avg_c, axis=1)
R_avg_std = np.std(R_avg_c, axis=1)

plt.figure(4)
for i in xrange(lam_c.shape[0]):
    plt.plot(R_avg_mean[0, :])
    plt.fill_between(R_avg_mean[0, :] + 2 * R_avg_std[0, :], R_avg_mean[0, :] - 2 * R_avg_std[0, :],
                     alpha=0.5, label=str(lam_c[i]))

plt.show()
