import numpy as np,tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error,mean_squared_error
s=(100+np.sin(np.arange(500)*.1)*5).astype('float32'); X=np.array([s[i:i+20] for i in range(480)])[...,None]; y=s[20:]
m=tf.keras.Sequential([layers.LSTM(32,input_shape=(20,1)),layers.Dense(1)]); m.compile('adam','mse'); m.fit(X,y,epochs=3,verbose=1); pred=m.predict(X,verbose=0).ravel(); print('MAE:',mean_absolute_error(y,pred),'RMSE:',mean_squared_error(y,pred)**.5)
