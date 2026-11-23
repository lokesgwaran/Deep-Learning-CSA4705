import numpy as np,tensorflow as tf
from tensorflow.keras import layers
prices=np.linspace(100,150,500)+np.sin(np.arange(500)*.1)*3; s=(prices-prices.min())/(prices.max()-prices.min()); X=np.array([s[i:i+20] for i in range(480)])[...,None]; y=s[20:]
m=tf.keras.Sequential([layers.SimpleRNN(32,input_shape=(20,1)),layers.Dense(1)]); m.compile('adam','mse'); m.fit(X,y,epochs=3,verbose=1); print('Predicted normalized value:',m.predict(X[:1],verbose=0))
