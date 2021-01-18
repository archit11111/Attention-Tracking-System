import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import joblib


class KNN(object):
    def __init__(self,data):
        self.data = data
    
    def train(self):
        accuracy_list = []
        f1_scores = []
        roc_list = []
        df_train = self.data[:int(len(self.data)*0.8)]
        df_test = self.data[-(len(self.data) - int(len(self.data)*0.8)):]
        print(df_test)
        X_train = df_train.drop('y_output',axis=1)
        y_train = df_train['y_output']
        X_test = df_test.drop('y_output',axis=1)
        y_test = df_test['y_output']
        for i in range(1,30):
            KNeighbors = KNeighborsClassifier(n_neighbors=i)
            KNeighbors.fit(X_train, y_train) 
            y_predict = KNeighbors.predict(X_test)
            y_score = KNeighbors.predict_proba(X_test)[:,1]
            accuracy_list.append(accuracy_score(y_test, y_predict))
            f1_scores.append(metrics.f1_score(y_test, y_predict))
            roc_list.append(metrics.roc_auc_score(y_test, y_score))
        accuracy_list.index(max(accuracy_list))+1            
        KNeighbors = KNeighborsClassifier(n_neighbors=accuracy_list.index(max(accuracy_list))+1)
        KNeighbors.fit(X_train, y_train) 
        print(X_test)
        y_predict = KNeighbors.predict(X_test)
        print(y_predict)
        y_score = KNeighbors.predict_proba(X_test)[:,1]
        accuracy = accuracy_score(y_test, y_predict)
        f1_score = metrics.f1_score(y_test, y_predict)
        roc = metrics.roc_auc_score(y_test, y_score)
        print("Accuracy % = ",accuracy, "\nF1-score = ", f1_score, "\nROC = ", roc)
        # print(confusion_matrix(y_test, pred_KN))

        # joblib.dump(KNeighbors, 'trained_models/KNN.pkl') 
