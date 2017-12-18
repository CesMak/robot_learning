# DMP-based controller.
#
# DMPPARAMS is the struct containing all the parameters of the DMP.
#
# PSI is the vector of basis functions.
#
# Q and QD are the current position and velocity, respectively.

import numpy as np

def dmpCtl (dmpParams, psi_i, q, qd):
    w=dmpParams.w
    alpha=dmpParams.alpha
    beta=dmpParams.beta
    tau=dmpParams.tau
    goal=dmpParams.goal

    fw = np.dot(psi_i.T, w)
    qdd = tau**2*(alpha*(beta*(goal-q)-(qd/tau))+fw)

    return qdd
