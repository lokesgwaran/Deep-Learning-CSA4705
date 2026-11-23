import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
actual=np.array([10,12,15,17,20]); pred=np.array([11,11,14,18,19]); mse=mean_squared_error(actual,pred); print('MAE:',mean_absolute_error(actual,pred)); print('MSE:',mse); print('RMSE:',mse**.5)
