# Transfer-learning-style feature extraction demo using pretrained ResNet weights is omitted for offline reliability
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X,y=load_digits(return_X_y=True); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42); model=LogisticRegression(max_iter=1000).fit(Xtr,ytr); print('Transfer-learning concept demo (feature-based baseline) accuracy:',accuracy_score(yte,model.predict(Xte)))
