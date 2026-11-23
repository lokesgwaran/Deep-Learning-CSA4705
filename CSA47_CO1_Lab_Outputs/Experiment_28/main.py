from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,log_loss
X,y=make_classification(n_samples=800,n_features=12,n_informative=8,n_redundant=2,random_state=42);Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y);s=StandardScaler();Xtr=s.fit_transform(Xtr);Xte=s.transform(Xte)
for act in ['relu','logistic','tanh']:
 m=MLPClassifier(hidden_layer_sizes=(32,16),activation=act,max_iter=500,early_stopping=True,random_state=42);m.fit(Xtr,ytr);prob=m.predict_proba(Xte);p=m.predict(Xte);print('Activation:',act,'iterations:',m.n_iter_,'accuracy:',round(accuracy_score(yte,p),3),'log loss:',round(log_loss(yte,prob),3))
