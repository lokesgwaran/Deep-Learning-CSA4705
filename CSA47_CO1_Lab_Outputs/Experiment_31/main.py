import numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
np.random.seed(42);area=np.random.randint(700,3000,120);bed=np.random.randint(1,6,120);price=.05*area+5*bed+np.random.normal(0,5,120);X=np.column_stack((area,bed));Xtr,Xte,ytr,yte=train_test_split(X,price,test_size=.25,random_state=42);s=StandardScaler();Xtr=s.fit_transform(Xtr);Xte=s.transform(Xte);m=MLPRegressor(hidden_layer_sizes=(32,16),max_iter=2000,random_state=42);m.fit(Xtr,ytr);p=m.predict(Xte);print('Actual:',np.round(yte[:10],2));print('Predicted:',np.round(p[:10],2));print('MSE:',round(mean_squared_error(yte,p),3),'MAE:',round(mean_absolute_error(yte,p),3),'R2:',round(r2_score(yte,p),3));plt.scatter(yte,p);plt.xlabel('Actual price');plt.ylabel('Predicted price');plt.title('Actual vs Predicted House Prices');plt.savefig('output/plot.png');plt.show()
