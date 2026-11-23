from math import log2
labels=['Low','Low','Low','High','High','Low','High','Low']; p_low=labels.count('Low')/len(labels); p_high=labels.count('High')/len(labels); entropy=-(p_low*log2(p_low)+p_high*log2(p_high)); print('Entropy:',entropy)
