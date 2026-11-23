import numpy as np
x=np.array([[1,3,2,4],[5,6,1,2],[7,8,3,4],[2,1,5,6]])
def pool(a,mode):
 out=[]
 for i in range(0,a.shape[0],2):
  row=[]
  for j in range(0,a.shape[1],2):
   block=a[i:i+2,j:j+2]; row.append(block.max() if mode=='max' else block.mean())
  out.append(row)
 return np.array(out)
print('Max pool:\\n',pool(x,'max')); print('Average pool:\\n',pool(x,'avg'))
