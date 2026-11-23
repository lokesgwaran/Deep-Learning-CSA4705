import numpy as np,tensorflow as tf
from tensorflow.keras import layers
np.random.seed(1); X=np.random.rand(500,10,3).astype('float32'); y=X[:,-1,0]
m=tf.keras.Sequential([layers.SimpleRNN(16,input_shape=(10,3)),layers.Dense(1)]); m.compile('adam','mse'); m.fit(X,y,epochs=3,verbose=1); print('Prediction:',m.predict(X[:1],verbose=0))
