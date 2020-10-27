import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import multilabel_confusion_matrix
from itertools import cycle
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE 
from imblearn.combine import SMOTEENN
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from scipy import stats
from sklearn.feature_selection import GenericUnivariateSelect, chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold

print('The scikit-learn version is {}.'.format(sk.__version__))
np.set_printoptions(suppress=True)
h = 10  # step size in the mesh


def KNN(X, y):
    clf = KNeighborsClassifier(n_neighbors=3)
    scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
    print(scores)
    y_pre = cross_val_predict(clf, X, y, cv=10)
    print(y_pre)
    # tn, fp, fn, tp = confusion_matrix(y, y_pre).ravel()
    print("KNN confusion matrix")
    # print((tn, fp, fn, tp))
    # print(recall_score(y, y_pre))
    # print(precision_score(y, y_pre))
    # print(accuracy_score(y, y_pre))
    print(scores.mean())
    return scores

def NB(X,y):
    clf = GaussianNB()
    scores = cross_val_score(clf, X, y, cv=10,scoring='accuracy')
    print(scores)
    y_pre = cross_val_predict(clf, X, y, cv=10)
    print(y_pre)
    tn, fp, fn, tp = confusion_matrix(y, y_pre).ravel()
    print("NB confusion matrix")
    print((tn, fp, fn, tp))
    print(recall_score(y, y_pre))
    print(precision_score(y, y_pre))
    print(accuracy_score(y, y_pre))
    print(scores.mean())
    return scores


def SVM(X,y):
    clf = svm.LinearSVC(C=20, dual=False)
    scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
    print(scores)
    y_pre = cross_val_predict(clf, X, y, cv=10)
    print(y_pre)
    # tn, fp, fn, tp = confusion_matrix(y, y_pre).ravel()
    print("SVM confusion matrix")
    # print((tn, fp, fn, tp))
    # print(recall_score(y, y_pre))
    # print(precision_score(y, y_pre))
    # print(accuracy_score(y, y_pre))
    print(scores.mean())
    return scores

def DT(X,y):
    clf = tree.DecisionTreeClassifier(min_samples_leaf=100, max_depth=6)
    scores = cross_val_score(clf, X, y, cv=10,scoring='accuracy')
    print(scores)
    y_pre = cross_val_predict(clf, X, y, cv=10)
    print(y_pre)
    tn, fp, fn, tp = confusion_matrix(y, y_pre).ravel()
    print("DT confusion matrix")
    print((tn, fp, fn, tp))
    print(recall_score(y, y_pre))
    print(precision_score(y, y_pre))
    print(accuracy_score(y, y_pre))
    print(scores.mean())
    return scores

def RF(X,y):
    clf = RandomForestClassifier(max_depth=6, random_state=0)
    scores = cross_val_score(clf, X, y, cv=10,scoring='accuracy')
    print(scores)
    y_pre = cross_val_predict(clf, X, y, cv=10)
    print(y_pre)
    # tn, fp, fn, tp = confusion_matrix(y, y_pre).ravel()
    print("RF confusion matrix")
    # print((tn, fp, fn, tp))
    # print(recall_score(y, y_pre))
    # print(precision_score(y, y_pre))
    # print(accuracy_score(y, y_pre))
    print(scores.mean())
    return scores

def ExtraTree(X,y):
    clf = ExtraTreesClassifier(max_depth=6, random_state=0)
    scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
    print(scores)
    y_pre = cross_val_predict(clf, X, y, cv=10)
    print(y_pre)
    tn, fp, fn, tp = confusion_matrix(y, y_pre).ravel()
    print("ExtraTree confusion matrix")
    print((tn, fp, fn, tp))
    print(recall_score(y, y_pre))
    print(precision_score(y, y_pre))
    print(accuracy_score(y, y_pre))
    print(scores.mean())
    return scores

if __name__ == "__main__":
    data = pd.read_csv(r"bank-additional-full.csv", sep=";")

    # Question 1
    # data_X = data.iloc[:, 0:20]
    # data_X = pd.get_dummies(data_X, columns=[
    #                         'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome'])
    # data_X['duration'] = np.log(data_X['duration']+1)
    # data_X['age'] = np.log(data_X['age']+1)
    # min_max_scaler = MinMaxScaler()
    # min_max_scaler.fit(data_X.values)
    # data_X = min_max_scaler.transform(data_X.values)
    # data_y = data.iloc[:, 20:21]
    # label_mapping = {'yes': 1, 'no': 0}
    # data_y = data_y['y'].map(label_mapping)
    # print("sample count before sampling:\n {}".format(data_y.value_counts()))
    # model_smote = SMOTE(random_state=42)
    # data_X_over_sampled,data_y_over_sampled = model_smote.fit_sample(data_X,data_y) 
    # print("sample count after over-sampling:\n {}".format(data_y_over_sampled.value_counts()))

    # undersample = NearMiss(version=1, n_neighbors=3)
    # data_X_under_sampled,data_y_under_sampled = undersample.fit_sample(data_X,data_y) 
    # print("sample count after under-sampling:\n {}".format(data_y_under_sampled.value_counts()))

    # smote_enn = SMOTEENN(random_state=42)
    # data_X_rampled,data_y_sampled = smote_enn.fit_sample(data_X,data_y) 
    # print("sample count after under-and-over-sampling:\n {}".format(data_y_sampled.value_counts()))

    # score0 = KNN(data_X, data_y)
    # score1 = KNN(data_X_over_sampled, data_y_over_sampled)
    # score2 = KNN(data_X_under_sampled, data_y_under_sampled)
    # score3 = KNN(data_X_rampled, data_y_sampled)

    # for i in range(10):
    #     print("Fold-"+str(i+1) + " & " + str(np.round(score1[i],4)) + " & " + str(np.round(score2[i],4)) +" & "+ str(np.round(score3[i],4))+" \\\\")

    # score0 = DT(data_X, data_y)
    # score1 = DT(data_X_over_sampled, data_y_over_sampled)
    # score2 = DT(data_X_under_sampled, data_y_under_sampled)
    # score3 = DT(data_X_rampled, data_y_sampled)

    # for i in range(10):
    #     print("Fold-"+str(i+1) + " & " + str(np.round(score1[i],4)) + " & " + str(np.round(score2[i],4)) +" & "+ str(np.round(score3[i],4))+" \\\\")

    # score0 = SVM(data_X, data_y)
    # score1 = SVM(data_X_over_sampled, data_y_over_sampled)
    # score2 = SVM(data_X_under_sampled, data_y_under_sampled)
    # score3 = SVM(data_X_rampled, data_y_sampled)

    # for i in range(10):
    #     print("Fold-"+str(i+1) + " & " + str(np.round(score1[i],4)) + " & " + str(np.round(score2[i],4)) +" & "+ str(np.round(score3[i],4))+" \\\\")

    # score0 = NB(data_X, data_y)
    # score1 = NB(data_X_over_sampled, data_y_over_sampled)
    # score2 = NB(data_X_under_sampled, data_y_under_sampled)
    # score3 = NB(data_X_rampled, data_y_sampled)

    # for i in range(10):
    #     print("Fold-"+str(i+1) + " & " + str(np.round(score1[i],4)) + " & " + str(np.round(score2[i],4)) +" & "+ str(np.round(score3[i],4))+" \\\\")
        
    # score0 = RF(data_X, data_y)
    # score1 = RF(data_X_over_sampled, data_y_over_sampled)
    # score2 = RF(data_X_under_sampled, data_y_under_sampled)
    # score3 = RF(data_X_rampled, data_y_sampled)

    # for i in range(10):
    #     print("Fold-"+str(i+1) + " & " + str(np.round(score1[i],4)) + " & " + str(np.round(score2[i],4)) +" & "+ str(np.round(score3[i],4))+" \\\\")

    # score0 = ExtraTree(data_X, data_y)
    # score1 = ExtraTree(data_X_over_sampled, data_y_over_sampled)
    # score2 = ExtraTree(data_X_under_sampled, data_y_under_sampled)
    # score3 = ExtraTree(data_X_rampled, data_y_sampled)

    # for i in range(10):
    #     print("Fold-"+str(i+1) + " & " + str(np.round(score1[i],4)) + " & " + str(np.round(score2[i],4)) +" & "+ str(np.round(score3[i],4))+" \\\\")

    # stats.ttest_rel(score61,score62)

    # transformer = GenericUnivariateSelect(chi2, mode='k_best', param=20)
    # X_new = transformer.fit_transform(data_X_rampled, data_y_sampled)
    # KNN(data_y_sampled, data_y)
    # sel = VarianceThreshold(threshold=0.2)
    # X_new = sel.fit_transform(data_X_rampled)
    # print(X_new.shape)
    # newscore1 = KNN(X_new, data_y_sampled)
    # newscore2 = SVM(X_new, data_y_sampled)
    # for i in range(10):
    #     print(str(np.round(newscore1[i],4)) + " & " + str(np.round(newscore2[i],4)) )

    # lr = LogisticRegression(penalty="l1",solver='liblinear',dual=True).fit(data_X, data_y)
    # model = SelectFromModel(lr, prefit=True)
    # X_new2 = model.transform(data_X_rampled)
    # print(X_new2.shape)
    # newscore1 = KNN(X_new2, data_y_sampled)
    # newscore2 = SVM(X_new2, data_y_sampled)
    # for i in range(10):
    #     print(str(np.round(newscore1[i],4)) + " & " + str(np.round(newscore2[i],4)) )

    #Question 2
    labor_neg_data = pd.read_csv(r"labor-neg.data")
    iris_data = pd.read_csv(r"iris.data")
    voting_data = pd.read_csv(r"house-votes-84.data")

    labor_X=labor_neg_data.iloc[:, 0:16]
    labor_y=labor_neg_data.iloc[:, 16:17]

    labor_X = pd.get_dummies(labor_X, columns=['cola', 'pension', 'educ_allw', 'vacation', 'lngtrm_disabil', 'dntl_ins', 'bereavement', 'empl_hplan'])
    label_mapping = {'good': 1, 'bad': 0}
    labor_y = labor_y['y'].map(label_mapping)
    labor_X = labor_X.replace('?', np.nan)
    labor_X = labor_X.T.fillna(labor_X.mean(axis=1)).T
    score1 = SVM(labor_X,labor_y)
    score2 = KNN(labor_X,labor_y)
    score3 = RF(labor_X,labor_y)
    for i in range(10):
        print(str(np.round(score1[i],4)) + " & " + str(np.round(score2[i],4)) +" & "+ str(np.round(score3[i],4))+" \\\\")

    print(score1.mean())
    print(score2.mean())
    print(score3.mean())
    print(score1.std())
    print(score2.std())
    print(score3.std())

    label_mapping = {'Iris-setosa': 1,
                     'Iris-versicolor': 0, 'Iris-virginica': 2}
    iris_X=iris_data.iloc[:, 0:4]
    iris_y=iris_data.iloc[:, 4:5]
    iris_y = iris_y['y'].map(label_mapping)

    score1 = SVM(iris_X,iris_y)
    score2 = KNN(iris_X,iris_y)
    score3 = RF(iris_X,iris_y)
    for i in range(10):
        print(str(np.round(score1[i],4)) + " & " + str(np.round(score2[i],4)) +" & "+ str(np.round(score3[i],4))+" \\\\")
    print(score1.mean())
    print(score2.mean())
    print(score3.mean())
    print(score1.std())
    print(score2.std())
    print(score3.std())


    label_mapping = {'republican': 1,'democrat': 0}
    voting_X=voting_data.iloc[:, 1:17]
    voting_y=voting_data.iloc[:, 0:1]
    voting_y = voting_y['y'].map(label_mapping)
    voting_X = pd.get_dummies(voting_X)

    score1 = SVM(voting_X,voting_y)
    score2 = KNN(voting_X,voting_y)
    score3 = RF(voting_X,voting_y)

    for i in range(10):
        print(str(np.round(score1[i],4)) + " & " + str(np.round(score2[i],4)) +" & "+ str(np.round(score3[i],4))+" \\\\")
    print(score1.mean())
    print(score2.mean())
    print(score3.mean())
    print(score1.std())
    print(score2.std())
    print(score3.std())
    plt.show()
