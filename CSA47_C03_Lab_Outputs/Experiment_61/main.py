import numpy as np,tensorflow as tf
from tensorflow.keras import layers
s=np.sin(np.arange(400)*.1).astype('float32'); X=np.array([s[i:i+15] for i in range(385)])[...,None]; y=s[15:]
for lr,units,batch in [(0.001,16,32),(0.01,32,32),(0.001,64,64)]:
 m=tf.keras.Sequential([layers.LSTM(units,input_shape=(15,1)),layers.Dense(1)]); m.compile(tf.keras.optimizers.Adam(lr),'mse'); m.fit(X,y,epochs=2,batch_size=batch,verbose=0); print(lr,units,batch,'loss:',m.evaluate(X,y,verbose=0))
