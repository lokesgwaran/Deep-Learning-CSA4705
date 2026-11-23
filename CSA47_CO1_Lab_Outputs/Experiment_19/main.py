import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import mean_squared_error,r2_score
X,y=make_regression(n_samples=250,n_features=6,noise=20,random_state=7);Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42);s=StandardScaler();Xtr=s.fit_transform(Xtr);Xte=s.transform(Xte)
for n,m in {'Linear':LinearRegression(),'L1 Lasso':Lasso(alpha=2),'L2 Ridge':Ridge(alpha=2)}.items():
 m.fit(Xtr,ytr);p=m.predict(Xte);print(n,'coefficients',np.round(m.coef_,2),'MSE',round(mean_squared_error(yte,p),2),'R2',round(r2_score(yte,p),3))
