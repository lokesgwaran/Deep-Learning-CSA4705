import numpy as np,tensorflow as tf
from tensorflow.keras import layers
data=np.random.rand(500,3).astype('float32'); X=np.array([data[i:i+10] for i in range(490)]); y=data[10:,0]
m=tf.keras.Sequential([layers.GRU(32,input_shape=(10,3)),layers.Dense(1)]); m.compile('adam','mse'); m.fit(X,y,epochs=3,verbose=1); print('Weather forecast:',m.predict(X[:1],verbose=0))
