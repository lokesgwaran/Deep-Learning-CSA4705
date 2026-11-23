import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
X,y=make_moons(n_samples=500,noise=.2,random_state=42);Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42);s=StandardScaler();Xtr=s.fit_transform(Xtr);Xte=s.transform(Xte)
for rate in [.001,.01,.1]:
 m=MLPClassifier(hidden_layer_sizes=(30,),solver='sgd',learning_rate_init=rate,max_iter=500,random_state=42);m.fit(Xtr,ytr);print('Rate',rate,'iterations',m.n_iter_,'loss',round(m.loss_,4),'accuracy',round(m.score(Xte,yte),3));plt.plot(m.loss_curve_,label=str(rate))
plt.xlabel('Iteration');plt.ylabel('Loss');plt.legend();plt.savefig('output/loss.png');plt.show()
