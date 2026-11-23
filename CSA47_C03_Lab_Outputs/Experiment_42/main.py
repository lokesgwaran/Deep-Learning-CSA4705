import tensorflow as tf
from tensorflow.keras import layers,models
(x,y),(xt,yt)=tf.keras.datasets.mnist.load_data(); x=x[...,None]/255.; xt=xt[...,None]/255.
for filters,k in [(16,3),(32,3),(32,5)]:
 m=models.Sequential([layers.Conv2D(filters,k,activation='relu',input_shape=(28,28,1)),layers.MaxPooling2D(),layers.Flatten(),layers.Dense(10,activation='softmax')]); m.compile('adam','sparse_categorical_crossentropy',['accuracy']); m.fit(x,y,epochs=1,validation_split=.1,verbose=0); print(filters,k,m.evaluate(xt,yt,verbose=0))
