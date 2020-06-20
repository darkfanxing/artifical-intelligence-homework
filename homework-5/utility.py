import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def split_data(data, label):
    x_train = np.concatenate((data[0:25, :], data[50:75, :], data[100:125, :]), axis=0)
    x_test = np.concatenate((data[25:50, :], data[75:100, :], data[125:150, :]), axis=0)

    y_train = np.concatenate((label[0:25], label[50:75], label[100:125]), axis=0)
    y_test = np.concatenate((label[25:50], label[75:100], label[125:150]), axis=0)

    return x_train, y_train, x_test, y_test


def knn_model(x_train, y_train, x_test, y_test):
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(x_train, y_train)

    return knn_model.score(x_test, y_test)


def get_model_accuracy(x_train, y_train, x_test, y_test):
    accuracy_1 = knn_model(x_train, y_train, x_test, y_test)
    
    x_train, x_test = x_test, x_train
    y_train, y_test = y_test, y_train
    accuracy_2 = knn_model(x_train, y_train, x_test, y_test)

    accuracy = (accuracy_1 + accuracy_2) / 2
    return accuracy