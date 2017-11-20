#Homework 1.3 d) code snipped of calculating the mean.
# This function is called within a for loop.

def calc_rmse(y_true,y_pred):
	rmse_t =np.power(y_true - y_pred,2)
	return np.sqrt(np.mean(rmse_t))
