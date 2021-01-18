# import pandas as pd
# import numpy as np
# from logistic_regression_model import LRModel
# from KNN import KNN
# from Random_Forest import RandomForest
# from Decision_Tree import DecisionTree
# from Naive_Bayes import NaiveBayes

# df1 = pd.read_csv('dset1.csv')
# df1['y_output'] = pd.Series(np.array([1]*len(df1)), index=df1.index)
# df1 = df1.drop(df1.columns[0], axis=1)
# df2 = pd.read_csv('dset2.csv')
# df2['y_output'] = pd.Series(np.array([0]*len(df2)), index=df2.index)
# df2 = df2.drop(df2.columns[0], axis=1)
# df = df1.append(df2, ignore_index=True)
# df = df.sample(frac=1).reset_index(drop=True)
# df.to_csv('test_data.csv')
# # print(df)




# dataset = pd.read_csv('test_data.csv')
# dataset = dataset.drop(dataset.columns[0:2],axis=1)
# # dataset = dataset.drop(dataset.columns[1:3],axis=1)

# # print(dataset)
# model = DecisionTree(dataset)
# model.train()

from datetime import datetime
import pytz
print(datetime.now(pytz.timezone('Asia/Kolkata')))
