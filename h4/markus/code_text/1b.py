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

        if i>1:
            if np.abs(R_avg[i] - R_avg[i-1]) < 1e-3:
                break
    print 'end'
    return Mu_w, Sigma_w, R_avg
