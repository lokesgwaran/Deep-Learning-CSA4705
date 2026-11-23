import numpy as np
text='we learn deep learning we train neural networks'; words=text.split(); vocab=sorted(set(words)); print('Vocabulary:',vocab); print('Next-word pairs:'); print(list(zip(words[:-1],words[1:])))
