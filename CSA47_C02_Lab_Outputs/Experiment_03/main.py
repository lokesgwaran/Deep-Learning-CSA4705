import numpy as np
A=np.array([[3,3],[2,5]]); det=np.linalg.det(A)
print('Determinant:',round(det)); print('Rank:',np.linalg.matrix_rank(A))
if abs(det)>1e-9: print('Inverse:\\n',np.linalg.inv(A)); print('Suitable (invertible): Yes')
else: print('Suitable (invertible): No')
