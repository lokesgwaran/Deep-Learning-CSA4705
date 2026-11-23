import numpy as np,tensorflow as tf
from tensorflow.keras import layers
demand=(100+20*np.sin(np.arange(600)*2*np.pi/24)+np.random.randn(600)*2).astype('float32'); X=np.array([demand[i:i+24] for i in range(576)])[...,None]; y=demand[24:]
m=tf.keras.Sequential([layers.SimpleRNN(32,input_shape=(24,1)),layers.Dense(1)]); m.compile('adam','mse'); m.fit(X,y,epochs=3,verbose=1); print('Demand prediction:',m.predict(X[:1],verbose=0))
