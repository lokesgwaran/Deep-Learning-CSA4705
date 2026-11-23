import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data=pd.DataFrame({'area':[1000,1200,1500,1800,2000,2200,900,1300,1700,2100,1600,2400], 'bedrooms':[2,2,3,3,4,4,1,2,3,4,3,5], 'location':['A','A','B','B','C','C','A','B','C','C','B','C'], 'price':[35,40,55,62,78,85,30,48,72,82,58,95]})
X=data[['area','bedrooms','location']]; y=data.price
prep=ColumnTransformer([('loc',OneHotEncoder(handle_unknown='ignore'),['location'])],remainder='passthrough')
model=make_pipeline(prep,LinearRegression()); model.fit(X,y); pred=model.predict(X)
print('Actual:',y.to_list()); print('Predicted:',pred.round(2).tolist())
print('MAE:',round(mean_absolute_error(y,pred),3),'MSE:',round(mean_squared_error(y,pred),3),'RMSE:',round(mean_squared_error(y,pred)**.5,3),'R2:',round(r2_score(y,pred),3))
