import numpy as np,tensorflow as tf
from tensorflow.keras import layers
s=np.sin(np.arange(400)*.1).astype('float32'); X=np.array([s[i:i+15] for i in range(385)])[...,None]; y=s[15:]
for layer in [layers.SimpleRNN(24),layers.LSTM(24),layers.GRU(24)]:
 m=tf.keras.Sequential([layer,layers.Dense(1)]); m.compile('adam','mse'); m.fit(X,y,epochs=2,verbose=0); print(layer.__class__.__name__,m.evaluate(X,y,verbose=0))
