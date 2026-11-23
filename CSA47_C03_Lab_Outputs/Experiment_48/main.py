import numpy as np,tensorflow as tf
from tensorflow.keras import layers
s=np.sin(np.arange(1000)*.04).astype('float32'); X=np.array([s[i:i+20] for i in range(len(s)-20)])[...,None]; y=s[20:]
m=tf.keras.Sequential([layers.SimpleRNN(32,input_shape=(20,1)),layers.Dense(1)]); m.compile('adam','mse'); m.fit(X,y,epochs=3,verbose=1); print('Forecast:',m.predict(X[:1],verbose=0))
