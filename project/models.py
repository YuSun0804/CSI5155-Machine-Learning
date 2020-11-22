from sklearn import svm
from sklearn.svm import LinearSVC
import sklearn as sk
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from sklearn.semi_supervised import LabelPropagation,LabelSpreading

print('The scikit-learn version is {}.'.format(sk.__version__))

def KNN(X_train, X_test, y_train, y_test):
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    y_pre = clf.predict(X_test)
    matrix = multilabel_confusion_matrix(y_test, y_pre)
    # print("DT confusion matrix")
    # print(matrix)
    print(recall_score(y_test, y_pre, average=None))
    print(precision_score(y_test, y_pre, average=None))
    print(accuracy_score(y_test, y_pre))

def NB(X_train, X_test, y_train, y_test):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pre = clf.predict(X_test)
    matrix = multilabel_confusion_matrix(y_test, y_pre)
    # print("DT confusion matrix")
    # print(matrix)
    print(recall_score(y_test, y_pre, average=None))
    print(precision_score(y_test, y_pre, average=None))
    print(accuracy_score(y_test, y_pre))


def DT(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pre_train = clf.predict(X_train)
    y_pre = clf.predict(X_test)
    matrix = multilabel_confusion_matrix(y_test, y_pre)
    # print("DT confusion matrix")
    # print(matrix)
    print(recall_score(y_test, y_pre, average=None))
    print(precision_score(y_test, y_pre, average=None))
    print(accuracy_score(y_test, y_pre))
    print(accuracy_score(y_train, y_pre_train))

def SVM(X_train, X_test, y_train, y_test):
    clf = svm.LinearSVC(C=20, dual=False)
    clf.fit(X_train, y_train)
    y_pre_train = clf.predict(X_train)
    y_pre = clf.predict(X_test)
    matrix = multilabel_confusion_matrix(y_test, y_pre)
    # print("DT confusion matrix")
    # print(matrix)
    print(recall_score(y_test, y_pre, average=None))
    print(precision_score(y_test, y_pre, average=None))
    print(accuracy_score(y_test, y_pre))
    print(accuracy_score(y_train, y_pre_train))

def RF(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pre_train = clf.predict(X_train)
    y_pre = clf.predict(X_test)
    matrix = multilabel_confusion_matrix(y_test, y_pre)
    # print("DT confusion matrix")
    # print(matrix)
    print(recall_score(y_test, y_pre, average=None))
    print(precision_score(y_test, y_pre, average=None))
    print(accuracy_score(y_test, y_pre))
    print(accuracy_score(y_train, y_pre_train))


def train_predict(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    y_pre_train = clf.predict(X_train)
    y_pre = clf.predict(X_test)
    matrix = multilabel_confusion_matrix(y_test, y_pre)
    # print("DT confusion matrix")
    # print(matrix)
    print(recall_score(y_test, y_pre, average=None))
    print(precision_score(y_test, y_pre, average=None))
    print(accuracy_score(y_test, y_pre))
    print(accuracy_score(y_train, y_pre_train))

def plot_roc(labels, predict_prob, title):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels, predict_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.figure()
    plt.title('ROC of '+title)
    plt.plot(false_positive_rate, true_positive_rate,
             'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')

def maskData(true_labels, mask_percentage):
    mask = np.zeros((1,len(true_labels)),dtype=bool)[0]
    labels = true_labels.copy()
    for l in np.unique(true_labels):
        deck = np.argwhere(true_labels.to_numpy() == l).flatten()        
        random.shuffle(deck)
        mask[deck[:int(mask_percentage * len(true_labels[true_labels == l]))]] = True
        labels[labels == l] = l
    labels[mask] = -1 
    return pd.Series(labels)



if __name__ == "__main__":
    data = pd.read_csv(r"data_fe.csv", sep=",")
    data_X = data.iloc[:, 1:]
    data_y = data.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.33, random_state=42)

    # supervised learning
    supervised_clfs={}
    # clfs['RF'] = RandomForestClassifier()
    supervised_clfs['SVM'] = svm.LinearSVC(dual=False)
    # clfs['DT'] = DecisionTreeClassifier()
    # clfs['NB'] = GaussianNB()
    # clfs['KNN'] = KNeighborsClassifier(n_neighbors=3)
    
    # for k,v in supervised_clfs.items():
    #     print('------------'+k+'--------------')
    #     train_predict(v, X_train, X_test, y_train, y_test)

    # semi-supervised learning
    semi_supervised_clfs = {}
    semi_supervised_clfs['LP'] = LabelPropagation(kernel='knn')
    semi_supervised_clfs['LS'] = LabelSpreading(kernel='knn')
    
    mask_percentage = [0, 0.1, 0.2, 0.5, 0.9, 0.95]
    for k,v in semi_supervised_clfs.items():
        for p in mask_percentage:
            print('------------'+k+'('+str(p)+')-------------')
            y_train = maskData(y_train, p)
            train_predict(v, X_train, X_test, y_train, y_test)
