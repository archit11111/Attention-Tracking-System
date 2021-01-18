from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import joblib



class DecisionTree(object):
    def __init__(self,data):
        self.data = data    
    
    def train(self):
        accuracy_list=[]
        max_depth = []
        df_train = self.data[:int(len(self.data)*0.8)]
        df_test = self.data[-(len(self.data) - int(len(self.data)*0.8)):]
        print(df_test)
        X_train = df_train.drop('y_output',axis=1)
        y_train = df_train['y_output']
        X_test = df_test.drop('y_output',axis=1)
        y_test = df_test['y_output']        
        for i in range(2,10):
            DecisionTree = DecisionTreeClassifier(max_depth = i)
            DecisionTree.fit(X_train, y_train)
            y_predict = DecisionTree.predict(X_test)
            accuracy_list.append(accuracy_score(y_predict, y_test))
            max_depth.append(i)
        
        best_depth = max_depth[accuracy_list.index(max(accuracy_list))]
        DecisionTree = DecisionTreeClassifier(max_depth=best_depth)
        DecisionTree.fit(X_train, y_train)
        y_predict = DecisionTree.predict(X_test)
        
        y_score = DecisionTree.predict_proba(X_test)[:,1]
        accuracy = accuracy_score(y_test, y_predict)
        f1_score = metrics.f1_score(y_test, y_predict)
        roc = metrics.roc_auc_score(y_test, y_score)
        print("Accuracy % = ",accuracy, "\nF1-score = ", f1_score, "\nROC = ", roc)
        print(confusion_matrix(y_test, y_predict))

        joblib.dump(DecisionTree, 'trained_models/DecisionTree.pkl') 
