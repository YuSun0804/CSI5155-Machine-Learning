from sklearn import svm
from sklearn.svm import LinearSVC
import sklearn as sk
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score,f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import multilabel_confusion_matrix,plot_confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from sklearn.semi_supervised import LabelPropagation,LabelSpreading
from itertools import cycle
from sklearn.model_selection import cross_val_predict

print('The scikit-learn version is {}.'.format(sk.__version__))


def train_predict(title, clf, data_X, data_y):
    print(title)

    y_pre = cross_val_predict(clf, data_X, data_y, cv=10)
    matrix = multilabel_confusion_matrix(data_y, y_pre)
    # disp = plot_confusion_matrix(clf, X_test, y_test,cmap=plt.cm.Blues)
    # disp.ax_.set_title("Confusion Matrix of " +title)
    print(matrix)
    print(recall_score(data_y, y_pre, average=None))
    print(negative_recall(matrix))
    print(f1_score(data_y, y_pre, average=None))

    print(accuracy_score(data_y, y_pre))

    # if( title == 'SVM'):
    #     y_pre_proba = clf.decision_function(X_test)
    #     y_test = pd.get_dummies(y_test)
    #     plot_roc(y_test, y_pre_proba, title)
    # else:
    #     y_pre_proba = clf.predict_proba(X_test)
    #     y_test = pd.get_dummies(y_test)
    #     plot_roc(y_test, y_pre_proba, title)

def negative_recall(matrix):
    negative_recall=[]
    for m in matrix:
        temp = m[0][0]/m.sum(axis=1)[0]
        negative_recall.append(temp)
    return negative_recall

def plot_roc(labels, predict_prob, title):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(predict_prob[0])):
        fpr[i], tpr[i], _ = roc_curve(labels[i], predict_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure()
    plt.title('ROC of '+ title)
    lw = 2
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(len(predict_prob[0])), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.4f})'
                 ''.format(i, roc_auc[i]))
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

    # supervised learning
    supervised_clfs={}
    supervised_clfs['Random Forest'] = RandomForestClassifier()
    # supervised_clfs['SVM'] = svm.LinearSVC(dual=False)
    # supervised_clfs['Decision Tree'] = DecisionTreeClassifier()
    # supervised_clfs['Naive Bayesian'] = GaussianNB()
    # supervised_clfs['KNN'] = KNeighborsClassifier(n_neighbors=3)
    
    for k,v in supervised_clfs.items():
        train_predict(k, v, data_X, data_y)

    # semi-supervised learning
    # semi_supervised_clfs = {}
    # semi_supervised_clfs['LP'] = LabelPropagation(kernel='knn')
    # semi_supervised_clfs['LS'] = LabelSpreading(kernel='knn')
    
    # mask_percentage = [0, 0.1, 0.2, 0.5, 0.9, 0.95]
    # for k,v in semi_supervised_clfs.items():
    #     for p in mask_percentage:
    #         print('------------'+k+'('+str(p)+')-------------')
    #         y_train = maskData(y_train, p)
    #         train_predict(v, X_train, X_test, y_train, y_test)
    plt.show()
