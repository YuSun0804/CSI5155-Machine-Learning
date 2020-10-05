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
# import graphviz
# from sklearn.tree import export_graphviz
# from sklearn.externals.six import StringIO
# import pydotplus
# import matplotlib.image as mpimg
# import io
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import multilabel_confusion_matrix
from itertools import cycle
from sklearn.metrics import roc_auc_score
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import MinMaxScaler

print('The scikit-learn version is {}.'.format(sk.__version__))
np.set_printoptions(suppress=True)
h = 10  # step size in the mesh


def KNN(X_train, X_test, y_train, y_test):
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    y_pre = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pre).ravel()
    print("KNN confusion matrix")
    print((tn, fp, fn, tp))
    print(recall_score(y_test, y_pre))
    print(precision_score(y_test, y_pre))
    print(accuracy_score(y_test, y_pre))

    y_pre_proba = clf.predict_proba(X_test)
    plot_roc(y_test, y_pre_proba[:, 1], "KNN")

    visualization(X_train, y_train, clf, "KNN")


def visualization(X_train, y_train, clf, title):
    clf.fit(X_train.iloc[:, 0:2], y_train)

    # Put the result into a color plot
    x_min, x_max = X_train.iloc[:, 0].min() - 1, X_train.iloc[:, 0].max() + 1
    y_min, y_max = X_train.iloc[:, 1].min() - 1, X_train.iloc[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    x = np.c_[xx.ravel(), yy.ravel()]
    # temp = np.zeros((x.shape[0],X_train.shape[1]-2))
    # Z = clf.predict(np.concatenate((x,temp),axis=1))
    Z = clf.predict(x)
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(['orange', 'cornflowerblue'])
    cmap_bold = ListedColormap(['darkorange', 'darkblue'])

    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Visualization of " + title)


def NB(X_train, X_test, y_train, y_test):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pre = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pre).ravel()
    print("NB confusion matrix")
    print((tn, fp, fn, tp))
    print(recall_score(y_test, y_pre))
    print(precision_score(y_test, y_pre))
    print(accuracy_score(y_test, y_pre))
    for index in range(len(X_train.columns.values)):
        print(str(X_train.columns.values[index]) + "&" + str(np.round(
            clf.theta_[0][index], 4))+"&" + str(np.round(clf.theta_[1][index], 4))+"&" + str(np.round(clf.sigma_[1][index], 4))+"&" + str(np.round(clf.sigma_[1][index], 4)))
    y_pre_proba = clf.predict_proba(X_test)
    plot_roc(y_test, y_pre_proba[:, 1], "Naive Bayesian")

    visualization(X_train, y_train, clf, "Naive Bayesian")


def SVM(X_train, X_test, y_train, y_test):
    clf = svm.SVC(kernel='linear', C=20)
    clf.fit(X_train, y_train)
    y_pre = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pre).ravel()
    print("SVM confusion matrix")
    print((tn, fp, fn, tp))
    print(recall_score(y_test, y_pre))
    print(precision_score(y_test, y_pre))
    print(accuracy_score(y_test, y_pre))

    y_pre_proba = clf.decision_function(X_test)
    plot_roc(y_test, y_pre_proba, "SVM")

    for index in range(len(X_train.columns.values)):
        if(index<31):
            print(str(X_train.columns.values[index]) + "&" + str(np.round(clf.coef_[0][index],4))+"&"+str(X_train.columns.values[index+32]) + "&" + str(np.round(clf.coef_[0][index+32],4)))
        
       
    visualSvm(X_train, y_train, clf)


def visualSvm(X_train, y_train, clf):
    X_train = X_train.iloc[:, 0:2]
    clf = svm.LinearSVC(C=20)
    clf.fit(X_train, y_train)
    decision_function = clf.decision_function(X_train)
    # we can also calculate the decision function manually
    # decision_function = np.dot(X_train, clf.coef_[0]) + clf.intercept_[0]
    support_vector_indices = np.where((2 * y_train - 1) * decision_function <= 1)[0]
    support_vectors = X_train.iloc[support_vector_indices]
    plt.figure()
    # plot the decision function
    title = ("Decision surface of linear SVC")
    # Set-up grid for plotting.
    plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1],
                c=y_train, s=30, cmap=plt.cm.Paired)
    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 10),
                         np.linspace(ylim[0], ylim[1], 10))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # plot decision boundary and margins
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                linestyles=['--', '-', '--'])
    # plot support vectors
    plt.scatter(support_vectors.iloc[:, 0], support_vectors.iloc[:, 1], s=100,
                linewidth=1, facecolors='none', edgecolors='k')


def DT(X_train, X_test, y_train, y_test):
    clf = tree.DecisionTreeClassifier(min_samples_leaf=100, max_depth=6)
    clf.fit(X_train, y_train)
    y_pre = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pre).ravel()
    print("DT confusion matrix")
    print((tn, fp, fn, tp))
    print(recall_score(y_test, y_pre))
    print(precision_score(y_test, y_pre))
    print(accuracy_score(y_test, y_pre))

    y_pre_proba = clf.predict_proba(X_test)
    plot_roc(y_test, y_pre_proba[:, 1], "Decision Tree")

    fig, ax = plt.subplots(figsize=(20, 10))  # whatever size you want
    tree.plot_tree(clf, filled=True)


def DT2(X_train, X_test, y_train, y_test):
    clf = tree.DecisionTreeClassifier(min_samples_leaf=100, max_depth=5)
    clf.fit(X_train, y_train)
    y_pre = clf.predict(X_test)
    matrix = multilabel_confusion_matrix(y_test, y_pre)
    print("DT confusion matrix")
    print(matrix)
    print(recall_score(y_test, y_pre, average=None))
    print(precision_score(y_test, y_pre, average=None))
    print(accuracy_score(y_test, y_pre))

    fig, ax = plt.subplots(figsize=(10, 10))  # whatever size you want
    tree.plot_tree(clf, filled=True)

    y_pre_proba = clf.predict_proba(X_test)
    y_test = pd.get_dummies(y_test, columns=['poutcome'])

    plot_roc2(y_test, y_pre_proba, "Decision Tree")

    macro_roc_auc_ovo = roc_auc_score(y_test, y_pre_proba, multi_class="ovo",
                                      average="macro")

    weighted_roc_auc_ovo = roc_auc_score(y_test, y_pre_proba, multi_class="ovo",
                                         average="weighted")

    print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
          "(weighted by prevalence)"
          .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))


def plot_roc2(labels, predict_prob, title):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(predict_prob[0])):
        fpr[i], tpr[i], _ = roc_curve(labels[i], predict_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure()
    plt.title('ROC of '+title)
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


if __name__ == "__main__":
    data = pd.read_csv(r"bank-additional-full.csv", sep=";")
    # import pandas_profiling
    # pfr = pandas_profiling.ProfileReport(data)
    # pfr.to_file("./test.html")

    # Question 1
    data_X = data.iloc[:, 0:20]
    data_X = pd.get_dummies(data_X, columns=[
                            'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome'])
    # data_X['duration'] = np.log(data_X['duration']+1)
    # data_X['age'] = np.log(data_X['age']+1)
    # min_max_scaler = MinMaxScaler()
    # min_max_scaler.fit(data_X.values)
    # data_X = min_max_scaler.transform(data_X.values)
    data_y = data.iloc[:, 20:21]
    label_mapping = {'yes': 1, 'no': 0}
    data_y = data_y['y'].map(label_mapping)

    X_train, X_test, y_train, y_test = train_test_split(
        data_X, data_y, test_size=0.33, random_state=42)
    DT(X_train, X_test, y_train, y_test)
    SVM(X_train, X_test, y_train, y_test)
    NB(X_train, X_test, y_train, y_test)
    KNN(X_train, X_test, y_train, y_test)

    # Question 2
    data_X1 = data.iloc[:, 0:14]
    data_X2 = data.iloc[:, 15:20]
    data_X = pd.concat([data_X1, data_X2], axis=1)
    data_X = pd.get_dummies(data_X, columns=[
                            'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week'])

    data_y = data.iloc[:, 14:15]
    label_mapping = {'failure': 0, 'nonexistent': 1, "success": 2}
    data_y = data_y['poutcome'].map(label_mapping)

    X_train, X_test, y_train, y_test = train_test_split(
        data_X, data_y, test_size=0.33, random_state=42)
    DT2(X_train, X_test, y_train, y_test)

    plt.show()
