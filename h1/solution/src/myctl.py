
# my_ctl.py computes the torques u vector with the given values!

import numpy as np

Kp = np.array([[60,30]])
Ki = np.array([[0.1,0.1]])
Kd = np.array([[10,6]])

def my_ctl(ctl, q, qd, q_des, qd_des, 
    qdd_des, q_hist, q_deshist, gravity, coriolis, M):
    u = np.zeros((2, 1))
    e = (q_des-q)
    ed = (qd_des-qd)
    ei=q_deshist-q_hist

    if ctl == 'P':
        u=Kp*e
    elif ctl == 'PD':
        u = Kp*e+Kd*ed
    elif ctl == 'PID':
        e_int = np.sum(ei,0)
        u = Kp*e+Kd*ed+Ki*e_int
    elif ctl == 'PD_Grav':
        u = Kp*e+Kd*ed+gravity
    elif ctl == 'ModelBased':
        qddref = qdd_des+Kd*ed+Kp*e
        u = np.dot(qddref,M)+coriolis+gravity
    return u.T
