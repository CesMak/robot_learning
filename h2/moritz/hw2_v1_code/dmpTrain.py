# Learns the weights for the basis functions.
#
# Q_IM, QD_IM, QDD_IM are vectors containing positions, velocities and
# accelerations of the two joints obtained from the trajectory that we want
# to imitate.
#
# DT is the time step.
#
# NSTEPS are the total number of steps.

from getDMPBasis import *
import numpy as np
from numpy.linalg import inv

class dmpParams():
    def __init__(self):
        self.alphaz = 0.0
        self.alpha = 0.0
        self.beta = 0.0
        self.Ts = 0.0
        self.tau = 0.0
        self.nBasis = 0.0
        self.goal = 0.0
        self.w = 0.0


def dmpTrain (q, qd, qdd, dt, nSteps):

    params = dmpParams()
    #Set dynamic system parameters
    params.alphaz = 3/(nSteps * dt -dt)
    params.alpha  = 25
    params.beta	  = 6.25
    params.Ts     = nSteps * dt -dt
    params.tau    = 1
    params.nBasis = 50
    params.goal   = np.array([[0.3, -0.8]])

    Phi = getDMPBasis(params, dt, nSteps)

    #Compute the forcing function
    # ft = qdd/(params.tau**2) + params.alpha*(qd/params.tau - params.beta*(params.goal.T-q))
    ft = qdd/(params.tau**2) - params.alpha*(params.beta*(params.goal.T-q) - qd/params.tau)
    #Learn the weights
    sigma = 1e-8
    pseudo_inv = inv((np.dot(Phi.T, Phi) + sigma*np.eye(Phi.shape[1])))
    params.w = np.dot(np.dot(pseudo_inv, Phi.T),ft.T)

    return params
