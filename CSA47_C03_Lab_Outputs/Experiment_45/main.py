import tensorflow as tf
from tensorflow.keras import layers
(x,y),(xt,yt)=tf.keras.datasets.mnist.load_data(); x=x[...,None]/255.; xt=xt[...,None]/255.
aug=tf.keras.Sequential([layers.RandomRotation(.1),layers.RandomZoom(.1)])
print('Original batch shape:',x[:8].shape); print('Augmented batch shape:',aug(x[:8]).shape); print('Image augmentation demo completed')
