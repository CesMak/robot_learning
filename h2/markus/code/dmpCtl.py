def dmpCtl (dmpParams, psi_i, q, qd):
    ...
    fw = np.dot(psi_i.T, w)
    qdd = tau**2*(alpha*(beta*(goal-q)-(qd/tau))+fw)
    return qdd
