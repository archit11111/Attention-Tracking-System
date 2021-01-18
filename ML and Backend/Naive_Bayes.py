from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import joblib

class NaiveBayes(object):
    def __init__(self,data):
        self.data = data    
    
    def train(self):
        df_train = self.data[:int(len(self.data)*0.9)]
        df_test = self.data[-(len(self.data) - int(len(self.data)*0.9)):]
        print(df_test)
        X_train = df_train.drop('y_output',axis=1)
        y_train = df_train['y_output']
        X_test = df_test.drop('y_output',axis=1)
        y_test = df_test['y_output']   
        NaiveB = GaussianNB()
        NaiveB.fit(X_train, y_train)
        y_predict = NaiveB.predict(X_test)
    
        y_score = NaiveB.predict_proba(X_test)[:,1]
        accuracy = accuracy_score(y_test, y_predict)
        f1_score = metrics.f1_score(y_test, y_predict)
        roc = metrics.roc_auc_score(y_test, y_score)
        print("Accuracy % = ",accuracy, "\nF1-score = ", f1_score, "\nROC = ", roc)
        print(confusion_matrix(y_test, y_predict))

        joblib.dump(NaiveB, 'trained_models/NaiveB.pkl') 
