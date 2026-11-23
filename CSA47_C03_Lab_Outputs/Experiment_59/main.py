import numpy as np,matplotlib.pyplot as plt
values=np.array([0.5**t for t in range(30)]); plt.plot(values,marker='o'); plt.title('Gradient magnitude across time steps'); plt.xlabel('Time step'); plt.ylabel('Gradient magnitude'); plt.grid(); plt.savefig('output/plot.png'); plt.show(); print('Final magnitude:',values[-1])
