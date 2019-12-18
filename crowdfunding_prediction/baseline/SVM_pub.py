import numpy as np
import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import svm

# from data_preprocessing.preprocessing import *
project_film_finally = pd.read_csv('./project_film_finally.txt')

start_time = dt.datetime.now()

train = project_film_finally.iloc[:, :156]
print(train.shape)

target = project_film_finally['result']
print(target.shape)

X_train, X_test, y_train, y_test = train_test_split(train, target, test_size = 0.2, random_state = 0)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# 不用交叉验证
clf = svm.SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)
clf_predict = clf.predict(X_test)
print("----clf-score--"*8, clf.score(X_test, y_test), dt.datetime.now())
print("----clf-accuracy--"*8, classification_report(y_test, clf_predict))


# 使用交叉验证
from sklearn.model_selection import cross_val_score

SVM_model = svm.SVC(kernel='linear', C=1)
score = cross_val_score(SVM_model, train, target, cv=5)
print(dt.datetime.now(), "----SVM-accuracy-cv--"*8, score)

# # 分类信息
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(SVM_model, train, target, cv=5)
print(dt.datetime.now(), "----accuracy-score---"*8, metrics.accuracy_score(target, predicted))
print(dt.datetime.now(), "----clf_report---"*8, classification_report(target, predicted))
print(dt.datetime.now(), "Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))

score = cross_val_score(SVM_model, train, target, cv=5, scoring='f1_macro')
print(dt.datetime.now(), "----SVM-score-f1--"*8, score)

from sklearn.model_selection import ShuffleSplit
n_samples = train.shape[0]
cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
print(cross_val_score(clf, train, target, cv=cv))

end_time = dt.datetime.now()
print("----- Time continues -----: ", end_time - start_time)
