from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
actual=np.array([0,1,1,0,1,0,1,0]); pred=np.array([0,1,0,0,1,1,1,0]); print(confusion_matrix(actual,pred)); print(classification_report(actual,pred))
