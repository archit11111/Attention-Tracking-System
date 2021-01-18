import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics

class LRModel(object):
    def __init__(self,data):
        self.data = data
    
    def train(self):
        df_train = self.data[:int(len(self.data)*0.8)]
        df_test = self.data[-(len(self.data) - int(len(self.data)*0.8)):]
        X_train = df_train.drop('y_output',axis=1)
        Y_train = df_train['y_output']
        X_test = df_test.drop('y_output',axis=1)
        y_test = df_test['y_output']
        lr = LogisticRegression().fit(X_train, Y_train)
        y_predict = lr.predict(X_test)
        print(y_predict)
        y_score_3 = lr.predict_proba(X_test)[:,1]
        acc3 = accuracy_score(y_test, y_predict)
        f1_score_3 = metrics.f1_score(y_test, y_predict)
        roc_3 = metrics.roc_auc_score(y_test, y_score_3)
        print([acc3,f1_score_3,roc_3])
        print(confusion_matrix(y_test, y_predict))
