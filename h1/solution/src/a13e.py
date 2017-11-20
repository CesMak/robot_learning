# A1.3e Model Selection

rmse_val = []

for i in nlist:
    # calculate the RMSE for n=1...9 of the training set
    theta = calc_theta(i,x_train,y_train)
    y_val = calc_y(x_val,i,theta)

    rmse_val.append(calc_rmse(Y_val,y_val))
