import numpy as np
from sklearn.neural_network import MLPClassifier
X=np.array([[0,0],[0,1],[1,0],[1,1]]);y=np.array([0,1,1,0]);m=MLPClassifier(hidden_layer_sizes=(4,),activation='tanh',solver='lbfgs',max_iter=5000,random_state=1);m.fit(X,y);print('Input and predicted XOR output:')
for row,p in zip(X,m.predict(X)):print(row,'->',p)
print('Training accuracy:',m.score(X,y))
