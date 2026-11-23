import numpy as np
P=np.array([[0,.5,.5],[1,0,0],[0,1,0]],float); values,vectors=np.linalg.eig(P)
print('Eigenvalues:',values); idx=np.argmax(np.real(values)); print('Dominant eigenvector:',np.real(vectors[:,idx]))
