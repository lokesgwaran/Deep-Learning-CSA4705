import numpy as np, matplotlib.pyplot as plt
np.random.seed(42); samples=np.random.normal(25,4,1000)
print('Sample mean:',samples.mean(),'Sample SD:',samples.std()); plt.hist(samples,bins=30); plt.title('Generated temperature samples'); plt.xlabel('Temperature'); plt.ylabel('Frequency'); plt.savefig('output/plot.png'); plt.show()
