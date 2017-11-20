#my_tastspace.py
import numpy as np
from math import pi

def my_taskSpace_ctl(ctl, dt, q, qd, gravity, coriolis,
 M, J, cart, desCart, resting_pos=None):
    KP = np.diag([60, 30])
    KD = np.diag([10, 6])
    gamma = 0.6
    dFact = 1e-6

    if ctl == 'JacTrans':
        #see 91 in script: qdot = gamma * J^T e 
        qd_des = gamma * J.T * (desCart - cart)
        error = q + qd_des * dt - q
        errord = qd_des - qd
        #u=M qdotdotref +c +g  
        #qdotdotref = KD*(qdotdes - qdot)+KP*(q+qd_des*dt -q)
        u = M * np.vstack(np.hstack([KP,KD])) * np.vstack([error,errord]) + coriolis + gravity
    elif ctl == 'JacPseudo':
        #see 91: qdot = eta*J^T Jpseudo *e 
        qd_des = gamma * J.T * np.linalg.pinv(J * J.T) * (desCart - cart)
        error = q + qd_des * dt - q
        errord = qd_des - qd
        u = M * np.vstack(np.hstack([KP,KD])) * np.vstack([error,errord]) + coriolis + gravity
    elif ctl == 'JacDPseudo':
        #Is this the damped Pseudoinverse?
        #see 97
        qd_des = J.T * np.linalg.pinv(J * J.T + dFact * np.eye(2)) * (desCart - cart)
        error = q + qd_des * dt - q
        errord = qd_des - qd
        u = M * np.vstack(np.hstack([KP,KD])) * np.vstack([error,errord]) + coriolis + gravity
    elif ctl == 'JacNullSpace':
        #See 94: qdot = Jpseudo xdot +(I-Jpseudo J)*KP(qrest-q)
        #See 97!
        qrest=resting_pos
        qq0=KP*(qrest-q)
        pseudo=J.T *np.linalg.inv(J*J.T+dFact*np.eye(len(J)))
        qd_des = pseudo*(desCart-cart)+(np.eye(len(J))-pseudo*J)*qq0
        error = q + qd_des * dt - q
        errord = qd_des - qd
        u = M * np.vstack(np.hstack([KP,KD])) * np.vstack([error,errord]) + coriolis + gravity
    return u

