import numpy as np
def sigmoid(x):return 1/(1+np.exp(-x))
x=np.array([[.5],[.8]]);target=np.array([[1.]]);W1=np.array([[.2,.4],[.3,.1]]);b1=np.zeros((2,1));W2=np.array([[.6,.7]]);b2=np.zeros((1,1));lr=.5;z1=W1@x+b1;a1=sigmoid(z1);z2=W2@a1+b2;a2=sigmoid(z2);loss=.5*(target-a2)**2;d2=(a2-target)*a2*(1-a2);dW2=d2@a1.T;db2=d2;d1=(W2.T@d2)*a1*(1-a1);dW1=d1@x.T;db1=d1;W2-=lr*dW2;b2-=lr*db2;W1-=lr*dW1;b1-=lr*db1;print('Loss before update:',float(loss));print('Updated W1:
',W1);print('Updated W2:
',W2);print('Updated output:',sigmoid(W2@sigmoid(W1@x+b1)+b2))
