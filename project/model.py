from numpy.lib.shape_base import tile
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
from sklearn.metrics import multilabel_confusion_matrix,plot_confusion_matrix,ConfusionMatrixDisplay,confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from sklearn.semi_supervised import LabelPropagation,LabelSpreading
from itertools import cycle
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold,learning_curve
import seaborn as sns
from scipy import interp
import time
from sklearn.linear_model import LogisticRegression

print('The scikit-learn version is {}.'.format(sk.__version__))

def train_predict_cv(title, clf, data_X, data_y):
    print(title)
    global recall_scores, negative_recalls, confusion_matrixs , f1_scores, accuracy_scores,predict_probs,y_tests,training_times,test_times
    recall_scores = []
    negative_recalls=[]
    confusion_matrixs = []
    f1_scores = []
    accuracy_scores = []
    predict_probs = []
    y_tests = []
    training_times = []
    test_times = []
    cv = StratifiedKFold(n_splits=10)
    for (train, test) in cv.split(data_X, data_y):
        train_predict(title, clf, data_X.iloc[train],data_X.iloc[test], data_y.iloc[train],data_y.iloc[test])
        print("--------------")
    # plot_learning_curve(clf, title, data_X, data_y, ylim=(0.2, 1.01), cv=cv, n_jobs=4)
    print(recall_scores)
    print(negative_recalls)
    print(f1_scores)
    print(accuracy_scores)
    print(np.average(accuracy_scores))
    print(training_times)
    print(test_times)
    y_test = pd.get_dummies(flatten(y_tests))
    plot_roc(y_test, flatten(predict_probs), title)
    # plot_confuse_matrix(np.average(confusion_matrixs,axis=0).astype(int),title,['0','1'],plt.cm.Blues)
    plot_confuse_matrix(np.average(confusion_matrixs,axis=0).astype(int),title,['0','1','2'],plt.cm.Blues)

def self_training_fit(X_train, y_train):
    unlabeled_index = np.arange(0,len(y_train))
    labeled_index = unlabeled_index[y_train != -1]
    unlabeled_index = unlabeled_index[y_train == -1]
    X_unlabeled = X_train.iloc[unlabeled_index]
    X_train = X_train.iloc[labeled_index]
    y_train = y_train.iloc[labeled_index]
    clf={}
    while True:     
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        if X_unlabeled.size ==0 :
            break
        pred_probs = clf.predict_proba(X_unlabeled)
        preds = clf.predict(X_unlabeled)

        df_pred_prob = pd.DataFrame([])
        df_pred_prob['preds'] = preds
        df_pred_prob['prob_0'] = pred_probs[:,0]
        df_pred_prob['prob_1'] = pred_probs[:,1]
        df_pred_prob['prob_2'] = pred_probs[:,2]
        df_pred_prob.index = X_unlabeled.index
        
        high_prob = pd.concat([df_pred_prob.loc[df_pred_prob['prob_0'] > 0.9],
                            df_pred_prob.loc[df_pred_prob['prob_1'] > 0.9],
                            df_pred_prob.loc[df_pred_prob['prob_2'] > 0.9]],                         
                            axis=0)

        X_train = pd.concat([X_train, X_unlabeled.loc[high_prob.index]], axis=0)
        y_train = pd.concat([y_train, high_prob.preds])      
        
        X_unlabeled = X_unlabeled.drop(index=high_prob.index)

        if(len(high_prob) == 0):
            break
    return clf

def self_training(X_train, X_test, y_train, y_test):
    start = time.process_time()
    clf = self_training_fit(X_train,y_train)
    end = time.process_time()
    training_times.append(end-start)
    start = time.process_time()
    y_pre = clf.predict(X_test)
    end = time.process_time()
    test_times.append(end-start)
    cm = multilabel_confusion_matrix(y_test, y_pre)
    confusion_matrixs.append(confusion_matrix(y_test, y_pre))
    recall_scores.append(recall_score(y_test, y_pre, average=None).tolist())
    negative_recalls.append(negative_recall(cm))
    f1_scores.append(f1_score(y_test, y_pre, average=None).tolist())
    accuracy_scores.append(accuracy_score(y_test, y_pre))
    y_pre_proba = clf.predict_proba(X_test)
    predict_probs.append(y_pre_proba)
    y_tests.append(y_test)

mask_percentage = [0,0.1]

def train_predict_no_cv(title, clf,  X_train, X_test, y_train, y_test):
    print(title)
    global recall_scores, negative_recalls, confusion_matrixs , f1_scores, accuracy_scores,predict_probs,y_tests,training_times,test_times
    recall_scores = []
    negative_recalls=[]
    confusion_matrixs = []
    f1_scores = []
    accuracy_scores = []
    predict_probs = []
    y_tests = []
    training_times = []
    test_times = []
    for p in mask_percentage:
        y_train = maskData(y_train, p)
        if title == 'self_training':
            self_training(X_train,X_test,y_train,y_test)
        else:
            train_predict(title, clf, X_train, X_test, y_train, y_test)
    # plot_learning_curve(clf, title, data_X, data_y, ylim=(0.2, 1.01), cv=cv, n_jobs=4)
    print(recall_scores)
    print(negative_recalls)
    print(f1_scores)
    print(accuracy_scores)
    print(accuracy_scores)
    print(training_times)
    print(test_times)
    y_test = pd.get_dummies(flatten(y_tests))
    plot_roc(y_test, flatten(predict_probs), title)
    # plot_confuse_matrix(np.average(confusion_matrixs,axis=0).astype(int),title,['0','1'],plt.cm.Blues)
    plot_confuse_matrix(np.average(confusion_matrixs,axis=0).astype(int),title,['0','1','2'],plt.cm.Blues)

def flatten(list_of_lists):
    flattened = []
    for sublist in list_of_lists:
        for val in sublist:
            flattened.append(val)
    return np.array(flattened)


def plot_confuse_matrix(cm, title, display_labels, cmap):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=display_labels)
    fig = disp.plot(cmap=cmap)
    fig.ax_.set_title("Confuse Matrix of "+title) 

def train_predict(title, clf, X_train, X_test, y_train, y_test):
    start = time.process_time()
    clf.fit(X_train, y_train)
    end = time.process_time()
    training_times.append(end-start)
    start = time.process_time()
    y_pre = clf.predict(X_test)
    end = time.process_time()
    test_times.append(end-start)
    cm = multilabel_confusion_matrix(y_test, y_pre)
    confusion_matrixs.append(confusion_matrix(y_test, y_pre))
    recall_scores.append(recall_score(y_test, y_pre, average=None).tolist())
    negative_recalls.append(negative_recall(cm))
    f1_scores.append(f1_score(y_test, y_pre, average=None).tolist())
    accuracy_scores.append(accuracy_score(y_test, y_pre))

    if( title == 'SVM'):
        y_pre_proba = clf.decision_function(X_test)
    else:
        y_pre_proba = clf.predict_proba(X_test)
    predict_probs.append(y_pre_proba)
    y_tests.append(y_test)
    
    # X_test =X_test.reset_index(drop=True)
    # y_test= y_test.reset_index(drop=True)

    # index = np.arange(0,len(y_pre))
    # index = index[y_test == 2]

    # bad_cases = pd.concat([y_test.iloc[index], pd.Series(y_pre,name='y_true').iloc[index],X_test.iloc[index]] ,axis=1)
    # bad_cases.iloc[0:100].to_csv(r'badcase.csv', mode='w+', index=False)
    

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
        random.seed(mask_percentage)        
        random.shuffle(deck)
        mask[deck[:int(mask_percentage * len(true_labels[true_labels == l]))]] = True
        labels[labels == l] = l
    labels[mask] = -1 
    return pd.Series(labels)



if __name__ == "__main__":
    data = pd.read_csv(r"data_fe1.csv", sep=",")
    data_X = data.iloc[:, 1:]
    data_y = data.iloc[:, 0]
    print(data_y.value_counts())

    # # supervised learning
    # supervised_clfs={}
    # supervised_clfs['Random Forest'] = RandomForestClassifier()
    # supervised_clfs['SVM'] = svm.LinearSVC(dual=False)
    # supervised_clfs['Decision Tree'] = DecisionTreeClassifier()
    # supervised_clfs['Naive Bayesian'] = GaussianNB()
    # supervised_clfs['KNN'] = KNeighborsClassifier(n_neighbors=3,n_jobs=-1)
    
    # for k,v in supervised_clfs.items():
    #     train_predict_cv(k, v, data_X, data_y)

    # semi-supervised learning
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.33, random_state=42)
    semi_supervised_clfs = {}
    # semi_supervised_clfs['LP'] = LabelPropagation(kernel='knn')
    # semi_supervised_clfs['LS'] = LabelSpreading(kernel='knn')
    semi_supervised_clfs['self_training'] = {}

    for k,v in semi_supervised_clfs.items():
        train_predict_no_cv(k,v, X_train, X_test, y_train, y_test)

    plt.show()
