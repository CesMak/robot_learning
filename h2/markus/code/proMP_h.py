def proMP (nBasis, condition=False):
    ...
    # estimate weight vectors w for all demonstrations
    W = np.eye(Phi.shape[1])*1e-12
    pseudo_inv = inv(np.dot(Phi.T, Phi) + W)
    ws = np.dot(np.dot(pseudo_inv, Phi.T), q.T)

    # fit a Gaussian over all weight vectors
    mean_w = np.mean(ws, axis=1)
    cov_w = np.cov(ws)
    mean_traj = np.dot(Phi, mean_w)
    std_traj = np.diag(np.dot(np.dot(Phi, cov_w), Phi.T))
