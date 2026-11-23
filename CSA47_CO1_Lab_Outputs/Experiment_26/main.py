from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
iris=load_iris();Xtr,Xte,ytr,yte=train_test_split(iris.data,iris.target,test_size=.3,random_state=42,stratify=iris.target);s=StandardScaler();Xtr=s.fit_transform(Xtr);Xte=s.transform(Xte);m=MLPClassifier(hidden_layer_sizes=(10,8),max_iter=1500,random_state=42,early_stopping=True);m.fit(Xtr,ytr);p=m.predict(Xte);print('Accuracy:',accuracy_score(yte,p));print(classification_report(yte,p,target_names=iris.target_names));print('Confusion matrix:
',confusion_matrix(yte,p))
