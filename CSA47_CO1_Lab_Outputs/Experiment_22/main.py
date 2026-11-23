from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.stats import loguniform
from sklearn.metrics import accuracy_score
X,y=load_breast_cancer(return_X_y=True);Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y);pipe=make_pipeline(StandardScaler(),SVC())
g=GridSearchCV(pipe,{'svc__C':[.1,1,10],'svc__kernel':['rbf','linear']},cv=3);g.fit(Xtr,ytr);print('Grid best:',g.best_params_,'score',g.best_score_,'test',accuracy_score(yte,g.predict(Xte)))
r=RandomizedSearchCV(pipe,{'svc__C':loguniform(.01,100),'svc__kernel':['rbf','linear']},n_iter=8,cv=3,random_state=42);r.fit(Xtr,ytr);print('Random best:',r.best_params_,'score',r.best_score_,'test',accuracy_score(yte,r.predict(Xte)))
try:
 from skopt import BayesSearchCV
 from skopt.space import Real,Categorical
 b=BayesSearchCV(pipe,{'svc__C':Real(.01,100,prior='log-uniform'),'svc__kernel':Categorical(['rbf','linear'])},n_iter=8,cv=3,random_state=42);b.fit(Xtr,ytr);print('Bayes best:',b.best_params_,'score',b.best_score_,'test',accuracy_score(yte,b.predict(Xte)))
except ImportError: print('Bayesian search skipped: install scikit-optimize to enable it')
