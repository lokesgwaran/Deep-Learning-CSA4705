import numpy as np,tensorflow as tf
from tensorflow.keras import layers
data=np.random.randn(500,3).astype('float32'); X=np.array([data[i:i+10] for i in range(490)]); y=data[10:,0]
m=tf.keras.Sequential([layers.SimpleRNN(32,return_sequences=True,input_shape=(10,3)),layers.SimpleRNN(16),layers.Dense(1)]); m.compile('adam','mse'); m.fit(X,y,epochs=3,verbose=1); print('Output:',m.predict(X[:1],verbose=0))
