import numpy as np
def compose(left,right,W,b): return np.tanh(W@np.concatenate([left,right])+b)
np.random.seed(2); W=np.random.randn(2,4)*.2; b=np.zeros(2); leaf_pos=np.array([1.,0.]); leaf_neg=np.array([0.,1.]); root=compose(leaf_pos,leaf_pos,W,b); root_neg=compose(leaf_neg,leaf_neg,W,b); print('Positive-tree representation:',root); print('Negative-tree representation:',root_neg); print('Demo complete')
