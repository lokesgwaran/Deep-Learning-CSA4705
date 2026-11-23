import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix
X=np.array([[25,550],[30,580],[35,600],[40,620],[45,650],[50,680],[55,700],[60,720],[70,760],[80,790],[28,540],[38,610],[48,660],[65,740],[75,780]],float);y=np.array([0,0,0,0,1,1,1,1,1,1,0,0,1,1,1]);Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.3,random_state=42);s=StandardScaler();Xtr=s.fit_transform(Xtr);Xte=s.transform(Xte);w=np.zeros(2);bias=0;lr=.1
for ep in range(100):
 for xi,t in zip(Xtr,ytr):
  pred=int(np.dot(xi,w)+bias>=0);err=t-pred;w+=lr*err*xi;bias+=lr*err
p=(Xte@w+bias>=0).astype(int);print('Predictions:',p);print('Actual:',yte);print('Accuracy:',accuracy_score(yte,p));print('Confusion matrix:
',confusion_matrix(yte,p))
