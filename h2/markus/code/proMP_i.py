def proMP (nBasis, condition=False):
    ...
    #Conditioning
    if condition:
        y_d = 3
        Sig_d = 0.0002
        t_point = np.round(2300/2)

        Phi_t = np.reshape(Phi[:,t_point],(30,1))
        tmp = np.dot(cov_w, Phi_t) / (Sig_d + np.dot(Phi_t.T,np.dot(cov_w,Phi_t)))

        tmp = np.reshape(tmp,(30,1))
        cov_w_new = cov_w - np.dot(np.dot(tmp, Phi_t.T), cov_w)
        mean_w_new = mean_w + np.dot(tmp, y_d-np.dot(Phi_t.T, mean_w))
        mean_traj_new = np.dot(Phi.T, mean_w_new)
        std_traj_new = np.diag(np.dot(np.dot(Phi.T, cov_w_new), Phi))

      ...