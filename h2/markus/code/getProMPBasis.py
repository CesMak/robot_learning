def getProMPBasis(dt, nSteps, n_of_basis, bandwidth):

    time = np.arange(dt,nSteps*dt,dt)
    T = dt * nSteps - dt
    nBasis  = n_of_basis
    b = bandwidth

    C = np.zeros(nBasis) # Basis function centres
    H = np.zeros(nBasis) # Basis function bandwidths

    H[:] = b
    C = np.linspace(0-2*b, T+2*b, nBasis)

    Phi = np.zeros((nSteps, nBasis))

    for i in xrange(nSteps):
        for j in xrange(nBasis):
            Phi[i,j] = np.exp(-1/2*((time[i]-C[j])**2)/H[j])
        # normalize the basis functions
        Phi[i, :] = Phi[i, :]/np.sum(Phi[i,:])

    return Phi
