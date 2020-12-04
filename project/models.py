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

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

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
    training_times.append(end-start)
    # print(np.around(clf.feature_importances_,4))
    cm = multilabel_confusion_matrix(y_test, y_pre)
    confusion_matrixs.append(confusion_matrix(y_test, y_pre))
    recall_scores.append(recall_score(y_test, y_pre, average=None))
    negative_recalls.append(negative_recall(cm))
    f1_scores.append(f1_score(y_test, y_pre, average=None))
    accuracy_scores.append(accuracy_score(y_test, y_pre))

    if( title == 'SVM'):
        y_pre_proba = clf.decision_function(X_test)
    else:
        y_pre_proba = clf.predict_proba(X_test)
    predict_probs.append(y_pre_proba)
    y_tests.append(y_test)
    
    X_test =X_test.reset_index(drop=True)
    y_test= y_test.reset_index(drop=True)

    index = np.arange(0,len(y_pre))
    index = index[y_test == 2]

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

    # supervised learning
    supervised_clfs={}
    supervised_clfs['Random Forest'] = RandomForestClassifier()
    # supervised_clfs['SVM'] = svm.LinearSVC(dual=False)
    supervised_clfs['Decision Tree'] = DecisionTreeClassifier()
    # supervised_clfs['Naive Bayesian'] = GaussianNB()
    # supervised_clfs['KNN'] = KNeighborsClassifier(n_neighbors=100)
    
    for k,v in supervised_clfs.items():
        train_predict_cv(k, v, data_X, data_y)

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
