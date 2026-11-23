# Offline toy cat/dog image classification demo without external image files
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
X,y=make_classification(n_samples=500,n_features=20,n_informative=10,random_state=7); m=LogisticRegression(max_iter=500).fit(X,y)
print('Toy Cat/Dog classifier accuracy:',m.score(X,y)); print('Class labels: 0=cat-like, 1=dog-like (synthetic demonstration)')
