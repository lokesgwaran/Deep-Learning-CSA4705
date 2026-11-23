import numpy as np
a=np.array([2,4,6]); b=np.array([5,1,2]); print('Euclidean:',np.linalg.norm(a-b)); print('Manhattan:',np.sum(np.abs(a-b)))
