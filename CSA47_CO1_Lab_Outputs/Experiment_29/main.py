import numpy as np
def sigmoid(x):return 1/(1+np.exp(-x))
x=np.array([[.5],[.8]]);W1=np.array([[.2,.4],[.3,.1]]);b1=np.array([[.1],[.2]]);W2=np.array([[.6,.7]]);b2=np.array([[.1]]);z1=W1@x+b1;a1=sigmoid(z1);z2=W2@a1+b2;a2=sigmoid(z2);print('Hidden layer weighted input:
',z1);print('Hidden layer output:
',a1);print('Output weighted input:
',z2);print('Final output:
',a2)
