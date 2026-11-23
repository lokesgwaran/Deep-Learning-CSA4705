import numpy as np, matplotlib.pyplot as plt
from scipy.signal import convolve2d
img=np.array([[1,2,1,0],[0,1,2,1],[2,1,0,1],[1,0,1,2]],float); kernel=np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
feature=convolve2d(img,kernel,mode='valid'); print('Feature map:\\n',feature); plt.imshow(feature,cmap='gray'); plt.title('Convolution feature map'); plt.colorbar(); plt.savefig('output/plot.png'); plt.show()
