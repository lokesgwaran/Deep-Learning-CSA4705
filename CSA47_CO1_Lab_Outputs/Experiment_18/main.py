import numpy as np, matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
np.random.seed(42); X=np.linspace(0,1,50).reshape(-1,1); y=np.cos(1.5*np.pi*X).ravel()+np.random.normal(0,.15,50); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.3,random_state=42)
grid=np.linspace(0,1,200).reshape(-1,1)
for d in [1,3,5,12]:
 m=make_pipeline(PolynomialFeatures(d),LinearRegression());m.fit(Xtr,ytr); print('Degree',d,'Train MSE',round(mean_squared_error(ytr,m.predict(Xtr)),4),'Test MSE',round(mean_squared_error(yte,m.predict(Xte)),4));plt.plot(grid,m.predict(grid),label=f'Degree {d}')
plt.scatter(X,y,color='black',s=15);plt.legend();plt.title('Polynomial Regression');plt.savefig('output/plot.png');plt.show()
