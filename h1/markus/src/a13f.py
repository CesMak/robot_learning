# A1.3f Cross Validation
def calc_loo(x, y, nlist):
    # this function uses the LOO algorithm for a given number of features
    rmse = np.zeros((np.size(x), np.size(nlist)))

    for i in xrange(np.size(x)):
        # choose one element as validation data, rest is training data
        x_train = np.delete(x,i)
        y_train = np.delete(y,i)
        x_val = x[i]
        y_val = y[i]

        for pos,j in enumerate(nlist):
            # calculate the RMSE for the different numbers of features
            theta = calc_theta(j,x_train,y_train)
            y_pred = calc_y(x_val,j,theta)
            rmse[i,pos] = calc_rmse(y_val,y_pred)

    means = []
    variance = []

    for i in xrange(np.size(nlist)):
        # calculate the mean and variance
        means.append(np.mean(rmse[:,i]))
        variance.append(np.var(rmse[:,i]))

