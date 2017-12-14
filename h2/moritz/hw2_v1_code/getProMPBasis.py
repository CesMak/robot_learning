import numpy as np
import matplotlib.pyplot as plt

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
        Phi[i, :] = Phi[i, :]/np.sum(Phi[i,:])

    return Phi

if __name__=='__main__':
    dt = 0.002
    time = np.arange(dt,3,dt)
    nSteps = len(time)
    getProMPBasis(dt, nSteps, 30, 0.2)

    plt.show()
