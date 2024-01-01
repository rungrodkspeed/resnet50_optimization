import numpy as np

def confusion_matrix(actual, predict):
    predict = np.array(predict, dtype=np.int8)
    actual = np.array(actual, dtype=np.int8)
    
    M = len( np.unique(actual) )
    
    cm = np.zeros((M,M))
    for _ in range(len(actual)):
        
        cm[actual[_], predict[_]] += 1
        
    return cm

