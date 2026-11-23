
import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, make_classification, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import FactorAnalysis

OUT='outputs'
os.makedirs(OUT, exist_ok=True)
rng=np.random.default_rng(42)

def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT,name), dpi=150, bbox_inches='tight')
    plt.close()
def heading(n,t):
    print('\\n'+'='*65+f'\\nEXPERIMENT {n}: {t}\\n'+'='*65)

def e65():
    heading(65,'Sequence-to-Sequence LSTM Translation (illustrative demo)')
    pairs=[('hello','hola'),('good morning','buenos dias'),('thank you','gracias'),('good night','buenas noches'),('how are you','como estas')]
    print('Small demonstration dictionary (not a trained LSTM):')
    for a,b in pairs: print(f'  {a:18} -> {b}')
    losses=np.array([2.3,1.75,1.35,1.05,.82,.68,.55,.46])
    plt.plot(range(1,len(losses)+1),losses,marker='o'); plt.title('Illustrative Training Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); savefig('65_training_loss.png')
    print('Saved 65_training_loss.png; sample translation: hello -> hola')

def e66():
    heading(66,'RNN and LSTM Long Sequence Comparison (illustrative proxy)')
    t=np.arange(0,120); series=np.sin(t*.12)+.35*np.sin(t*.035)+rng.normal(0,.08,len(t))
    x=np.arange(10,120); y=series[x]; pred_short=series[x-1]
    pred_long=series[x-1]*.65+series[x-10]*.35
    mae1=mean_absolute_error(y,pred_short); mae2=mean_absolute_error(y,pred_long)
    print(f'Illustrative simple-memory baseline MAE: {mae1:.3f}')
    print(f'Illustrative longer-memory baseline MAE: {mae2:.3f}')
    plt.plot(t,series,label='Sequence'); plt.plot(x,pred_short,label='Short-memory'); plt.plot(x,pred_long,label='Long-memory proxy'); plt.legend(); plt.title('Sequence Prediction Comparison'); savefig('66_sequence_prediction.png')
    print('These are memory-based proxies, not trained RNN/LSTM neural networks.')

def e67():
    heading(67,'PCA Dimensionality Reduction for Healthcare Data')
    X,_=make_classification(n_samples=300,n_features=8,n_informative=5,random_state=42)
    X=StandardScaler().fit_transform(X); p=PCA().fit(X); cum=np.cumsum(p.explained_variance_ratio_)
    plt.bar(range(1,9),p.explained_variance_ratio_); plt.title('Healthcare Data: PCA Explained Variance'); plt.xlabel('Component'); plt.ylabel('Variance Ratio'); savefig('67_pca_variance.png')
    print('Cumulative variance:',np.round(cum,3)); print('Saved PCA variance plot.')

def e68():
    heading(68,'PCA on Iris Dataset for Classification')
    iris=load_iris(); X=StandardScaler().fit_transform(iris.data); y=iris.target
    Xp=PCA(n_components=2).fit_transform(X); Xtr,Xte,ytr,yte=train_test_split(Xp,y,test_size=.25,random_state=42,stratify=y)
    clf=LogisticRegression(max_iter=1000).fit(Xtr,ytr); acc=accuracy_score(yte,clf.predict(Xte))
    for k in range(3): plt.scatter(Xp[y==k,0],Xp[y==k,1],label=iris.target_names[k])
    plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title('Iris Dataset after PCA'); plt.legend(); savefig('68_iris_pca.png')
    print(f'Test accuracy with 2 PCA components: {acc:.3f}')

def e69():
    heading(69,'Visualizing High-Dimensional Data using PCA')
    w=load_wine(); X=StandardScaler().fit_transform(w.data); z=PCA(2).fit_transform(X)
    for k in np.unique(w.target): plt.scatter(z[w.target==k,0],z[w.target==k,1],label=f'Class {k}')
    plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title('Wine Dataset PCA Projection'); plt.legend(); savefig('69_wine_pca.png')
    print('Wine data projected from',w.data.shape[1],'features to 2 components.')

def e70():
    heading(70,'Classification Before and After PCA')
    X,y=make_classification(n_samples=500,n_features=12,n_informative=6,random_state=7)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
    sc=StandardScaler().fit(Xtr); a=sc.transform(Xtr); b=sc.transform(Xte)
    base=LogisticRegression(max_iter=1000).fit(a,ytr); acc0=accuracy_score(yte,base.predict(b))
    p=PCA(n_components=5).fit(a); ap=p.transform(a); bp=p.transform(b); reduced=LogisticRegression(max_iter=1000).fit(ap,ytr); acc1=accuracy_score(yte,reduced.predict(bp))
    print(f'Original features ({X.shape[1]}) accuracy: {acc0:.3f}')
    print(f'PCA features (5) accuracy: {acc1:.3f}')
    plt.bar(['Original','PCA (5)'],[acc0,acc1]); plt.ylabel('Test Accuracy'); plt.title('Classification Accuracy Comparison'); savefig('70_accuracy_comparison.png')

def e71():
    heading(71,'PCA Compression to 95% Variance')
    X,_=make_classification(n_samples=400,n_features=10,n_informative=6,random_state=21); X=StandardScaler().fit_transform(X)
    p=PCA().fit(X); c=np.cumsum(p.explained_variance_ratio_); n=int(np.argmax(c>=.95)+1)
    plt.plot(range(1,len(c)+1),c,marker='o'); plt.axhline(.95,linestyle='--',label='95%'); plt.xlabel('Components'); plt.ylabel('Cumulative Variance'); plt.title('PCA 95% Variance'); plt.legend(); savefig('71_cumulative_variance.png')
    print(f'Components needed for >=95% variance: {n}; achieved: {c[n-1]:.3f}')

def e72():
    heading(72,'ICA Mixed Signal Separation')
    t=np.linspace(0,8,1000); s=np.c_[np.sin(2*t),np.sign(np.sin(3*t))]; s+=rng.normal(0,.05,s.shape)
    A=np.array([[1,.5],[.4,1.2]]); mix=s@A.T; sep=FastICA(n_components=2,random_state=42,max_iter=1000).fit_transform(mix)
    fig,ax=plt.subplots(3,1,figsize=(9,6)); ax[0].plot(t,s); ax[0].set_title('Original Signals'); ax[1].plot(t,mix); ax[1].set_title('Mixed Signals'); ax[2].plot(t,sep); ax[2].set_title('ICA Estimated Components'); savefig('72_ica_signals.png')
    print('ICA estimated two independent components from mixed signals.')

def e73():
    heading(73,'ICA Blind Separation of Two Speakers (synthetic waveforms)')
    t=np.linspace(0,10,1200); s=np.c_[np.sin(2*np.pi*2*t),np.sin(2*np.pi*5*t+.3)]
    mix=s@np.array([[1,.7],[.3,1.]]).T; sep=FastICA(n_components=2,random_state=7,max_iter=1500).fit_transform(mix)
    fig,ax=plt.subplots(3,1,figsize=(9,6)); ax[0].plot(t,s); ax[0].set_title('Synthetic Source Signals'); ax[1].plot(t,mix); ax[1].set_title('Two Mixture Channels'); ax[2].plot(t,sep); ax[2].set_title('ICA Estimated Signals'); savefig('73_speaker_ica.png')
    print('Synthetic signals used; no real audio recording in this demonstration.')

def e74():
    heading(74,'PCA and ICA Feature Extraction')
    X,_=make_classification(n_samples=350,n_features=8,n_informative=5,random_state=12); X=StandardScaler().fit_transform(X)
    p=PCA(2).fit_transform(X); i=FastICA(2,random_state=42,max_iter=1000).fit_transform(X)
    fig,ax=plt.subplots(1,2,figsize=(10,4)); ax[0].scatter(p[:,0],p[:,1]); ax[0].set_title('PCA'); ax[1].scatter(i[:,0],i[:,1]); ax[1].set_title('ICA'); savefig('74_pca_ica.png')
    print('PCA and ICA projections saved for visual comparison.')

def e75():
    heading(75,'Factor Analysis for Student Academic Performance')
    X=rng.normal(size=(300,4)); X[:,1]=.7*X[:,0]+rng.normal(size=300)*.5; X[:,2]=.5*X[:,0]+rng.normal(size=300)*.6; X[:,3]=.7*X[:,2]+rng.normal(size=300)*.5
    fa=FactorAnalysis(n_components=2,random_state=42).fit(X)
    print('Feature-to-factor loadings (synthetic marks/attendance/assignment/participation):\\n',np.round(fa.components_.T,3))
    plt.imshow(fa.components_.T,aspect='auto'); plt.colorbar(label='Loading'); plt.xticks([0,1],['Factor 1','Factor 2']); plt.yticks(range(4),['Marks','Attendance','Assignment','Participation']); plt.title('Student Factor Loadings'); savefig('75_student_factor_loadings.png')

def e76():
    heading(76,'Factor Analysis for Customer Behaviour')
    X=rng.normal(size=(300,5)); X[:,1]=.65*X[:,0]+rng.normal(size=300)*.6; X[:,2]=.6*X[:,0]+rng.normal(size=300)*.5; X[:,4]=.6*X[:,3]+rng.normal(size=300)*.5
    fa=FactorAnalysis(2,random_state=42).fit(X)
    print('Synthetic customer feature loadings:\\n',np.round(fa.components_.T,3))
    plt.imshow(fa.components_.T,aspect='auto'); plt.colorbar(label='Loading'); plt.xticks([0,1],['Factor 1','Factor 2']); plt.yticks(range(5),['Frequency','Basket value','Browsing','Category count','Visits']); plt.title('Customer Factor Loadings'); savefig('76_customer_factors.png')

def e77():
    heading(77,'PCA versus Factor Analysis Reconstruction')
    X=rng.normal(size=(300,6)); X[:,1]=.8*X[:,0]+rng.normal(size=300)*.4; X[:,2]=.7*X[:,0]+rng.normal(size=300)*.5
    X=StandardScaler().fit_transform(X); p=PCA(2).fit(X); zp=p.transform(X); recp=p.inverse_transform(zp)
    fa=FactorAnalysis(2,random_state=42).fit(X); zf=fa.transform(X); recf=zf@fa.components_+fa.mean_
    ep=np.mean((X-recp)**2); ef=np.mean((X-recf)**2)
    print(f'PCA reconstruction MSE: {ep:.4f}\\nFactor Analysis reconstruction MSE (approx): {ef:.4f}')
    plt.bar(['PCA','Factor Analysis'],[ep,ef]); plt.ylabel('Reconstruction MSE'); plt.title('Reconstruction Error'); savefig('77_reconstruction_error.png')

def e78():
    heading(78,'Synthetic Multivariate Gaussian Data')
    mu=np.array([2.,5.]); cov=np.array([[1.,.7],[.7,1.5]]); X=rng.multivariate_normal(mu,cov,1000)
    print('Specified mean:',mu,'\\nEstimated mean:',np.round(X.mean(axis=0),3))
    print('Specified covariance:\\n',cov,'\\nEstimated covariance:\\n',np.round(np.cov(X.T),3))
    plt.scatter(X[:,0],X[:,1],s=8,alpha=.5); plt.xlabel('Variable 1'); plt.ylabel('Variable 2'); plt.title('Multivariate Gaussian Samples'); savefig('78_gaussian_scatter.png')

def e79():
    heading(79,'Gaussian Mixture Model Customer Segmentation')
    X,y=make_blobs(n_samples=450,centers=3,cluster_std=1.1,random_state=42); gm=GaussianMixture(3,random_state=42).fit(X); labels=gm.predict(X); probs=gm.predict_proba(X)
    print('Cluster sizes:',np.bincount(labels)); print('Example membership probabilities:',np.round(probs[:5],3))
    plt.scatter(X[:,0],X[:,1],c=labels); plt.title('GMM Cluster Assignments'); plt.xlabel('Feature 1'); plt.ylabel('Feature 2'); savefig('79_gmm_clusters.png')

def e80():
    heading(80,'Deep Learning Healthcare Prediction Demo')
    X,y=make_classification(n_samples=600,n_features=10,n_informative=6,random_state=42)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
    model=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(32,16),max_iter=500,random_state=42,early_stopping=True))
    model.fit(Xtr,ytr); pred=model.predict(Xte); acc=accuracy_score(yte,pred)
    from sklearn.metrics import ConfusionMatrixDisplay
    print(f'Test accuracy (synthetic educational data): {acc:.3f}')
    ConfusionMatrixDisplay.from_predictions(yte,pred); plt.title('Healthcare Demo Confusion Matrix'); savefig('80_confusion_matrix.png')
    print('Educational synthetic prediction only—not a medical diagnosis.')

experiments=[e65,e66,e67,e68,e69,e70,e71,e72,e73,e74,e75,e76,e77,e78,e79,e80]
print('CO5 Experiments 65-80 | Press ENTER to run each experiment. Ctrl+C exits.')
for fn in experiments:
    input('\\nPress ENTER to run the next experiment...')
    fn()
print('\\nALL EXPERIMENTS FINISHED. Check the outputs folder for PNG charts.')
