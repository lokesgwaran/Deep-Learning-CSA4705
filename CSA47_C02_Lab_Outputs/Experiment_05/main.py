import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
a=np.array([4,2,0]); b=np.array([3,1,1]); print('Dot product:',np.dot(a,b)); print('Cosine similarity:',cosine_similarity([a],[b])[0][0])
