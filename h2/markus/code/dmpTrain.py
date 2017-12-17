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
    ft = qdd/(params.tau**2) - params.alpha*(params.beta*(params.goal.T-q) - qd/params.tau)
    #Learn the weights
    sigma = 1e-8
    pseudo_inv = inv((np.dot(Phi.T, Phi) + sigma*np.eye(Phi.shape[1])))
    params.w = np.dot(np.dot(pseudo_inv, Phi.T),ft.T)

    return params
