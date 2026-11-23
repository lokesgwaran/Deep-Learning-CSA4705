import numpy as np
text='deep learning models learn patterns from data. '*20; chars=sorted(set(text)); lookup={c:i for i,c in enumerate(chars)}; seq=np.array([lookup[c] for c in text]); X=[]; y=[]
for i in range(len(seq)-10): X.append(seq[i:i+10]); y.append(seq[i+10])
print('Character vocabulary:',len(chars),'Training windows:',len(X)); print('Sample input sequence:',text[:11]); print('Next character:',text[10])
