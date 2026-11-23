from sklearn.metrics import precision_score,recall_score,f1_score
y_true=[0,1,1,2,0,2,1]; y_pred=[0,1,0,2,0,1,1]
print('Precision:',precision_score(y_true,y_pred,average='weighted')); print('Recall:',recall_score(y_true,y_pred,average='weighted')); print('F1:',f1_score(y_true,y_pred,average='weighted'))
