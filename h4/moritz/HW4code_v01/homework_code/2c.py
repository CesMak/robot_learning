numDim = 10
numSamples = 25
maxIter = 100
numTrials = 10

# Do your learning
def PolicyGradient(alpha = 0.1):
    ...
    for i in xrange(maxIter):
        ...

        temp = np.zeros(theta.shape)
        for j in xrange(numSamples):
            temp[j, :] = np.dot(np.linalg.inv(Sigma), theta[j] - Mu) * (R[j] - R_avg[i])
                # gradient
        grad_mu = 1./numSamples * np.sum(temp, axis=0)

        Mu = Mu + alpha*grad_mu
        ...
    print 'R_avg_end: {}'.format(R_avg[-1])
    return Mu, Sigma, R_avg
