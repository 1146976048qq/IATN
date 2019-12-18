# -*- coding: utf-8 -*-
# 导入所需的python库

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# from data_preprocessing.preprocessing import *

project_film_finally = pd.read_csv('/Users/kzhang/Desktop/IATN/Test/project_film_finally.txt')

train = project_film_finally.iloc[:, :]
target = project_film_finally['result']

X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.2, random_state=0)
y_test = pd.DataFrame(y_test)
y_train = pd.DataFrame(y_train)

y_train = y_train['result']
y_test = y_test['result']

LR_model = LogisticRegression(C = 1.0, penalty = 'l2')
LR_model.fit(X_train, y_train)

print(LR_model.score(X_test,y_test))

# 给出交叉验证集的预测结果，评估准确率、召回率、F1值
pred_test = LR_model.predict(X_test)
print(classification_report(y_test, pred_test))

from sklearn.model_selection import cross_val_score
cross_val = cross_val_score(LR_model, train, target, cv=3)
print(cross_val)

from sklearn import metrics
from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(LR_model, train, target, cv=10)
print(metrics.accuracy_score(target, predicted))
print(classification_report(target, predicted))

