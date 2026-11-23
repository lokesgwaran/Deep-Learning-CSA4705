from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
X,y=load_breast_cancer(return_X_y=True); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
s=StandardScaler(); Xtr=s.fit_transform(Xtr); Xte=s.transform(Xte)
for name,m in [('Large',MLPClassifier(hidden_layer_sizes=(100,100),max_iter=500,random_state=42)),('Improved',MLPClassifier(hidden_layer_sizes=(30,),max_iter=500,alpha=.01,early_stopping=True,random_state=42))]:
 m.fit(Xtr,ytr); print(name,'training accuracy:',accuracy_score(ytr,m.predict(Xtr)),'testing accuracy:',accuracy_score(yte,m.predict(Xte)))
