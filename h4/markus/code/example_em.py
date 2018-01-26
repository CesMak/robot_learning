from matplotlib import pyplot as plt
import numpy as np
from Pend2dBallThrowDMP import *
from math import pi, exp, sqrt
#%matplotlib inline

env = Pend2dBallThrowDMP()

# Theta sind die samples!

numDim = 10
numSamples = 25  #N samples
maxIter = 100
numTrials = 10

# @return: scalar
def update_mu(w, theta):
    # nimm erste spalte von theta * w1 und summiere dann über die spalte auf!
    return np.sum(w * theta, axis=0) / np.sum(w)

# @return: matrix
def update_cov(w, mu, theta):
    storage = np.zeros((numDim, numDim))
    for i in range(0,numDim):
        tmp = (theta[i,:] - mu).reshape((numDim,1))
        storage += w[i]*np.dot(tmp,tmp.T)
    return storage*1/np.sum(w)

def get_beta(R,lambda_):
    return lambda_/(np.amax(R)-np.amin(R))

def calc_w(vec_rewards,lambda_):
    beta = get_beta(vec_rewards,lambda_)
    w = np.zeros((numSamples,1))
    for i in range(0,numSamples):
        w[i] = exp( (vec_rewards[i]-np.amax(vec_rewards))*beta )
    return w

def check_converge(R_last, R_actual):
    tmp = abs(R_last-R_actual)
    #print(tmp)
    if(tmp<1e-3):
        return True
    else:
        return False

def one_iteration(Mu_w, Sigma_w):
    rewards = np.zeros((numSamples))
    samples = np.zeros((numSamples, numDim))
    for i in range(0, numSamples):
        samples[i,:]=np.random.multivariate_normal(Mu_w, Sigma_w)
        rewards[i] = env.getReward(samples[i,:])
    return [rewards, samples]

def EM(lambda_,name=""):
    #initialize the distribution:
    Mu_w = np.zeros(numDim)
    Sigma_w = np.eye(numDim) * 1e6
    rewards = np.zeros((numSamples,1))
    samples = np.zeros((numSamples, numDim))
    w = np.zeros((numSamples,1))
    R_actual_average = 0
    R_last_average = 0
    for i in range(100):
        [rewards, samples] = one_iteration(Mu_w, Sigma_w)
        w = calc_w(rewards,lambda_)
        Mu_w = update_mu(w, samples)
        Sigma_w = update_cov(w, Mu_w, samples)+np.eye((numDim))
        R_actual_average = np.mean(rewards)
        if(check_converge(R_last_average, R_actual_average)):
            break
        R_last_average=R_actual_average
    print("Last average",name," :", R_last_average)
    env.animate_fig(np.random.multivariate_normal(Mu_w, Sigma_w))
    str_ = 'EM_lambda='+str(lambda_)+" "+name+".pdf"
    plt.savefig(str_)
    return R_last_average

def EM(lam = 7):
    # For example, let initialize the distribution...
    Mu_w = np.zeros(numDim)
    Sigma_w = np.eye(numDim) * 1e6

    R_mean_old = 0
    # Do your learning
    R_avg = np.zeros(maxIter)
    for i in range(maxIter):
        # reward
        R = np.zeros((numSamples, 1))
        theta = np.zeros((numSamples, numDim))
        for j in range(numSamples):
            theta[j, :] = np.random.multivariate_normal(Mu_w, Sigma_w)
            R[j] = env.getReward(theta[j, :])
        R_avg[i] = np.mean(R)
        #print 'R_mean_{}: {}'.format(i, R_avg)

        # weights
        beta = lam / (np.max(R) - np.min(R))
        w = np.exp((R - np.max(R)) * beta)

        Mu_w = np.sum(w * theta, axis=0) / np.sum(w)

        temp = np.zeros((numDim, numDim))

        for j in range(numSamples):
            diff = (theta[j, :] - Mu_w).reshape((numDim, 1))
            temp += w[j] * np.dot(diff, diff.T)
        Sigma_w = temp / np.sum(w) + np.eye(numDim)

        if i>1:
            if np.abs(R_avg[i] - R_avg[i-1]) < 1e-3:
                break
    return Mu_w, Sigma_w, R_avg


def EM_1c(numlearning, lambdas):
    R_averages = np.zeros((numlearning,1))
    means_R = np.zeros((len(lambdas),1))
    stds_R = np.zeros((len(lambdas),1))
    for j in range(0,len(lambdas)):
        for i in range(0,numlearning):
             R_averages[i] = EM(lambdas[j])
        means_R[j] = np.mean(R_averages)
        stds_R[j] = np.std(R_averages)
    plt.plot(lambdas,means_R)
    plt.show()

#Moritz:
mu, sigma, r = EM()
env.animate_fig(np.random.multivariate_normal(mu, sigma))
plt.show()
print(r)

#HW4 Task 1b:
#EM(7, name="1b")

#HW4 Task 1c:
#lambdas = np.array([3,7,25])
#EM_1c(10, lambdas)