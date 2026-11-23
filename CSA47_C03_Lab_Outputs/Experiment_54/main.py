import numpy as np,tensorflow as tf
from tensorflow.keras import layers
s=(100+np.sin(np.arange(500)*.1)*20).astype('float32'); X=np.array([s[i:i+20] for i in range(480)])[...,None]; y=s[20:]
for layer in [layers.SimpleRNN(32),layers.LSTM(32)]:
 m=tf.keras.Sequential([layer,layers.Dense(1)]); m.compile('adam','mse'); m.fit(X,y,epochs=2,verbose=0); print(layer.__class__.__name__,'loss:',m.evaluate(X,y,verbose=0))
