import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
texts=['this is great','i love this','this is bad','i hate it','very nice','terrible product']; labels=[1,1,0,0,1,0]
v=CountVectorizer(); X=v.fit_transform(texts); m=LogisticRegression().fit(X,labels); print('Predictions:',m.predict(X)); print('Sentiment labels: 1 positive, 0 negative')
