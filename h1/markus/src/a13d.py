# A1.3d Training a Model
def calc_rmse(y_true, y_pred):
    # this function calculates the RMSE from given y-values
    rmse_t = np.power(y_true-y_pred,2)
    return np.sqrt(np.mean(rmse_t))


nlist = np.array(range(1,10))
rmse_train = []

for i in nlist:
    # calc the rmse for n=1...9
    theta = calc_theta(i,x_train,y_train)
    y_pred = calc_y(x_train,i,theta)

    rmse_train.append(calc_rmse(Y_train,y_pred))
