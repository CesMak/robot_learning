# A1.3g Kernel Regression
def kernel(xi,xj, sigma = 0.15):
    # this function calculates the kernel of the two x-values xi and xj
    return np.exp(-1/(sigma**2)*abs(xi-xj)**2)


def kernel_regression(x_train):
    # this function calculates the K-matrix for the kernel regression
    n = np.size(x_train)
    K = np.zeros((n, n))
    for i in xrange(n):
        for j in xrange(n):
            K[i, j] = kernel(x_train[i], x_train[j])
    return K


def kernel_predict(x, X, Y, K):
    # this function uses the K-matrix to predict new values
    f_x = np.zeros((np.size(x), 1))
    for i in xrange(np.size(x)):
        k = kernel(x[i],X)
        f_x[i,0] = np.dot(np.dot(k.T, inv(K)),Y)
    return f_x


K = kernel_regression(x_train)
# calculate the predicted values for the training data
y_pred_kernel = kernel_predict(x_plot, x_train, y_train, K)
# calculate the predicted values for the validation data
y_kernel_pred_val = kernel_predict(x_val, x_train, y_train, K)
# calculate the RMSE
kernel_rmse = calc_rmse(Y_val, y_kernel_pred_val)
