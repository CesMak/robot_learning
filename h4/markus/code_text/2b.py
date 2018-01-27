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
            temp[j, :] = np.dot(np.linalg.inv(Sigma), theta[j] - Mu) * R[j]
                # gradient
        grad_mu = 1./numSamples * np.sum(temp, axis=0)

        Mu = Mu + alpha*grad_mu

        if i>1:
            if np.abs(R_avg[i] - R_avg[i-1]) < 1e-3:
                break
    print 'R_avg_end: {}'.format(R_avg[-1])
    return Mu, Sigma, R_avg
