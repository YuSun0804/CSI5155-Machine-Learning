import numpy as np
import matplotlib.pyplot as plt
import os
import math

DECIMAL_POINT = 3  # float number sensitivities
MAX_ITERATION = 500  # iteration number of cluster calculation
CLUSTER_FOLDER_NAME = "kmeans"  # clustered files folder
CLUSTER_FILE_NAME = "clusters"  # file names
DATA_PATH = "polblogs.txt"     # data file
DATA_LABELS_PATH = "polblogs-labels.txt"       # label file
SIMILARITY_MATRIX = "blogsSimilarityMatrix.txt"  # similarity matrix name


# find manhattan distance similarities
def manhattan_distance(first_row, second_row):
    if np.array_equal(first_row,second_row):
        return 1
    counter_first = 0
    counter_second = 0
    counter_mutual = 0

    for i in range(len(first_row)):
        if first_row[i] == 1 and second_row[i] == 1:
            counter_mutual = counter_mutual + 1
        elif first_row[i] == 1 and second_row[i] == 0:
            counter_first += 1
        elif first_row[i] == 0 and second_row[i] == 1:
            counter_second += 1

    if(counter_first + counter_second - counter_mutual) == 0:
        return 0
    else:
        return counter_mutual / (counter_first + counter_second - counter_mutual)


def pre_process_data():
    print("Pre-processing data started...")
    data = np.loadtxt(DATA_PATH)
    x, y = data[:, 0], data[:,1]

    blogs = np.zeros((1218, 1218),dtype=int)

    for i in range(len(x)):
        a, b = int(x[i]), int(y[i])
        blogs[a,b] = 1
        blogs[b,a] = 1
    print("Connected adjency mtrix created!")
    print(blogs)
    # np.savetxt("blogsMatrix.txt", np.int_(blogs),fmt="%d")

    similarity_matrix = np.zeros((1218, 1218),dtype=float)

    print("Manhattan distance calculations started...")
    for i in range(1218):
        for j in range(1218):
            similarity = manhattan_distance(blogs[i,:], blogs[j,:])
            similarity_matrix[i, j] = similarity

    print("Similarity matrix created!")
    np.savetxt(SIMILARITY_MATRIX, similarity_matrix, fmt="%.3f", delimiter="\t")

    print("Pre processing finished, all files saved!!")


def kmeans(data, k, tour):
    print("K-Means Started...")
    # initial centroids
    not_converged = True
    centroids = np.zeros((len(data), k), dtype=float)
    for i in range(2):
        for j in range(len(centroids)):
            centroids[j,i] = np.around(np.random.uniform(0.0, 1.0),DECIMAL_POINT)

    cluster0 = []
    cluster1 = []
    counter = 0
    while not_converged:
        counter += 1
        if counter > MAX_ITERATION:
            not_converged = False
        # print("Iteration number = ", counter)
        tmp_cluster0 = cluster0[:]
        tmp_cluster1 = cluster1[:]
        for i in range(len(data)):
            sumC1 = 0
            sumC2 = 0
            for j in range(len(data)):
                sumC1 = sumC1 + abs(centroids[i,1] - data[i,j])
                sumC2 = sumC2 + abs(centroids[i,0] - data[i,j])
            if sumC1 < sumC2:
                if i not in cluster0:
                    cluster0.append(i)
                if i in cluster1:
                    cluster1.remove(i)
            else:
                if i not in cluster1:
                    cluster1.append(i)
                if i in cluster0:
                    cluster0.remove(i)
        if tmp_cluster0 == cluster0 and tmp_cluster1 == cluster1:
            print("Clusters Same Thats finished!")
            not_converged = False
        else:
            oldCentroids = np.copy(centroids)
            centroids = update_centroids(centroids,cluster0,cluster1,data)
            if np.array_equal(oldCentroids,centroids):
                print("Centroids same, Finished!")
                not_converged = False

    print("Final Iteration number = ", counter)
    print("cluster0 = ", cluster0)
    print("cluster1 = ", cluster1)
    labels = np.empty((len(data),2), dtype=int)
    for i in range(len(cluster0)):
        labels[cluster0[i],0] = cluster0[i]
        labels[cluster0[i],1] = 0

    for i in range(len(cluster1)):
        labels[cluster1[i],0] = cluster1[i]
        labels[cluster1[i],1] = 1

    if not os.path.exists(CLUSTER_FOLDER_NAME):
        os.makedirs(CLUSTER_FOLDER_NAME)
        print("Clusters folder created.")

    file_path = CLUSTER_FOLDER_NAME
    file_path += "/"
    filename = CLUSTER_FILE_NAME
    file_path += filename
    file_path += str(tour)
    file_path += ".txt"
    # filename += repr(tour)
    # filename += repr(".txt")

    np.savetxt(file_path, np.int_(labels), fmt="%d")
    print("K-Means clustering finished. Files are ready!")


def update_centroids(centroids,cluster0,cluster1,data):
    for i in range(len(data)):
        sums = 0
        for j in range(len(cluster0)):
            sums = sums + data[cluster0[j],i]
        centroids[i,0] = np.around(sums / len(cluster0),DECIMAL_POINT)

    for i in range(len(data)):
        sums = 0
        for j in range(len(cluster1)):
            sums = sums + data[cluster1[j],i]
        centroids[i,1] = np.around(sums / len(cluster1),DECIMAL_POINT)

    return centroids


# it returns best cluster for f_measure and purity calculations
def plot_kmeans():
    print("Plot Clusters...")
    polblogs_labels = np.int_(np.loadtxt(DATA_LABELS_PATH))
    folder_path = CLUSTER_FOLDER_NAME
    folder_path += "/"
    files = os.listdir(folder_path)
    accuracy_list = []

    # calculate accuracy for 50 results
    for name in files:
        preLabel = folder_path
        preLabel += name
        cluster_labels = np.int_(np.loadtxt(preLabel))
        cluster_labels_only = cluster_labels[:,1]
        matched = 0
        for i in range(len(polblogs_labels)):
            if polblogs_labels[i] != cluster_labels_only[i]:
                matched += 1

        if(len(polblogs_labels) - matched) > matched:
            accuracy = (len(polblogs_labels) - matched) / len(polblogs_labels)
            # print(accuracy)
            accuracy_list.append(accuracy)
        else:
            accuracy = matched/len(polblogs_labels)
            accuracy_list.append(accuracy)
            # print(accuracy)
    # print(accuracy_list.index(max(accuracy_list)))

    best_accuracy_file = files[accuracy_list.index(max(accuracy_list))]
    folder_path = CLUSTER_FOLDER_NAME
    folder_path += "/"
    folder_path += best_accuracy_file
    cluster_labels1 = np.int_(np.loadtxt(folder_path))
    list_of_symbols = ["o", "x", "v", "^", "<", "H", "8", "|", "_", "P"]

    # plot the best one
    for i in range(len(cluster_labels1)):
        if cluster_labels1[i,1] == 0:
            plt.scatter(cluster_labels1[i, 0], cluster_labels1[i, 1], color="red", marker=list_of_symbols[0])
        if cluster_labels1[i,1] == 1:
            plt.scatter(cluster_labels1[i, 0], cluster_labels1[i, 1], color="blue", marker=list_of_symbols[1])
    # plt.savefig('filename1.png', dpi=300)
    plt.show()
    plt.clf()

    # plot all results.
    # counter = 0
    # for name in files:
    #     counter += 1
    #     preLabel = "labels/"
    #     preLabel += name
    #     cluster_labels = np.int_(np.loadtxt(preLabel))
    #     print("\nF-Measures for: ", name)
    #     # f_measure(cluster_labels)
    #     for i in range(len(cluster_labels)):
    #         if cluster_labels[i, 1] == 0:
    #             plt.scatter(cluster_labels[i, 0], cluster_labels[i, 1], color="red", marker=list_of_symbols[0])
    #         if cluster_labels[i, 1] == 1:
    #             plt.scatter(cluster_labels[i, 0], cluster_labels[i, 1], color="blue", marker=list_of_symbols[1])
    #     png_name = "Graph"
    #     png_name += str(counter)
    #     png_name +=".png"
    #     plt.savefig(png_name, dpi=300)
    #     plt.show()
    #     plt.clf()

    return cluster_labels1


def f_measure(cluster):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    polblogs_labels = np.int_(np.loadtxt(DATA_LABELS_PATH))
    classlabels = cluster[:,1]

    for i in range(len(polblogs_labels)):
        if polblogs_labels[i] == 1 and polblogs_labels[i] == classlabels[i]:
            tp += 1
        elif polblogs_labels[i] == 0 and polblogs_labels[i] == classlabels[i]:
            tn += 1
        elif polblogs_labels[i] == 1 and polblogs_labels[i] != classlabels[i]:
            fn += 1
        elif polblogs_labels[i] == 0 and polblogs_labels[i] != classlabels[i]:
            fp += 1

    # k-means gives random class labels initially
    # it could be reversed from the original labels. If it is, then we need to change
    # all values
    if tp < fp:
        tmp = tp
        tp = fp
        fp =tmp
    if tn < fn:
        tmp = tn
        tn = fn
        fn = tmp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print("True positive counts = ", tp)
    print("True negative counts = ", tn)
    print("False positive counts = ", fp)
    print("False negative counts = ", fn)
    print("Precision = ", precision)
    print("Recall = ", recall)
    print("F-Measure = ", 2 * (precision * recall) / (precision + recall))
    entropy = purity((tp+fp)/len(polblogs_labels), (tn+fn)/len(polblogs_labels))
    print("Purity = ", entropy)


def purity(pos, neg):
    print("Calculating entropy")
    entropy = (-pos*math.log(pos)) - (-neg * math.log(neg))
    return entropy


def find_clusters():
    data = np.loadtxt(SIMILARITY_MATRIX)

    # run 50 times
    # creates files with index
    # for i in range(50):
    #     print("Tour = ", i)
    #     kmeans(data,2,i)
    kmeans(data, 2,1)


pre_process_data()
find_clusters()
cluster = plot_kmeans()
f_measure(cluster)

print("Finised")
