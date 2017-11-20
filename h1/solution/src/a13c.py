# A1.3c Linear Least Squares
def phi(x_t, n_t):
    # this function calculates the value of the feature function
    # x_t: x value, n_t number of features
    phi_t = np.zeros((n_t,np.size(x_t)))
    for i in xrange(n_t):
        phi_t[i]=np.sin(pow(2,i)*x_t)
    return phi_t

def calc_y(x_t, n_t, theta_t):
    # this function calculates the prediction y from given values for x and theta
    phi_t = phi(x_t,n_t)
    return np.dot(phi_t.T,theta_t)

def calc_theta(n,x_train,y_train):
    # this function calculates the theta parameter with the left pseudo-inverse
    N = np.size(x_train)
    Phi = np.zeros((N,n))
    Y = np.array([y_train])
    Y = Y.T

    for i in xrange(N):
        Phi[i,...] = phi(x_train[i], n).T

    # calculate left pseudo-inverse
    temp = np.dot(Phi.T, Phi)
    temp = np.dot(inv(temp), Phi.T)

    return np.dot(temp, Y)
