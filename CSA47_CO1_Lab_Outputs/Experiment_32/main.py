import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,log_loss
X,y=make_classification(n_samples=1000,n_features=10,n_informative=6,weights=[.9,.1],flip_y=.02,random_state=42);Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y);s=StandardScaler();Xtr=s.fit_transform(Xtr);Xte=s.transform(Xte);m=MLPClassifier(hidden_layer_sizes=(24,12),max_iter=1000,early_stopping=True,random_state=42);m.fit(Xtr,ytr);prob=m.predict_proba(Xte)[:,1];p=(prob>=.5).astype(int);eps=1e-15;cl=np.clip(prob,eps,1-eps);bce=-np.mean(yte*np.log(cl)+(1-yte)*np.log(1-cl));print('Binary cross-entropy:',round(bce,4));print('Log loss check:',round(log_loss(yte,prob),4));print('Accuracy:',round(accuracy_score(yte,p),4));print('Confusion matrix:
',confusion_matrix(yte,p));print(classification_report(yte,p))
