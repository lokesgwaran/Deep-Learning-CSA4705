import tensorflow as tf
from tensorflow.keras import layers,models
(x,y),(xt,yt)=tf.keras.datasets.mnist.load_data(); x=x[...,None]/255.; xt=xt[...,None]/255.
for bn in [False,True]:
 seq=[layers.Conv2D(32,3,input_shape=(28,28,1))];
 if bn: seq.append(layers.BatchNormalization())
 seq += [layers.Activation('relu'),layers.MaxPooling2D(),layers.Flatten(),layers.Dense(10,activation='softmax')]
 m=models.Sequential(seq); m.compile('adam','sparse_categorical_crossentropy',['accuracy']); m.fit(x,y,epochs=1,verbose=0); print('BatchNorm',bn,'test:',m.evaluate(xt,yt,verbose=0))
