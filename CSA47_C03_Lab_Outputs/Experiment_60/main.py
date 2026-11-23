import numpy as np,tensorflow as tf
from tensorflow.keras import layers
s=np.sin(np.arange(500)*.08).astype('float32')
for n in [5,10,20]:
 X=np.array([s[i:i+n] for i in range(len(s)-n)])[...,None]; y=s[n:]; m=tf.keras.Sequential([layers.SimpleRNN(24,input_shape=(n,1)),layers.Dense(1)]); m.compile('adam','mse'); m.fit(X,y,epochs=2,verbose=0); print('Sequence length',n,'loss:',m.evaluate(X,y,verbose=0))
