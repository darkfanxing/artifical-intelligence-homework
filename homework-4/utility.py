import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import scipy.linalg as la

def get_data_mean(data):
    data_number = data.shape[0]

    data_sum = np.zeros(shape=(1, data.shape[1]))
    for index in range(data_number):
        data_sum += data[index]
    
    return data_sum / data_number


def get_covariance(data):
    data_mean = get_data_mean(data)
    data_number = data.shape[0]
    feature_number = data.shape[1]

    covariance = np.zeros(shape=(feature_number, feature_number))
    for index in range(data_number):
        separability = data[index] - data_mean
        covariance += (separability).dot(separability.T)

    return covariance / data_number


def get_eigenvalue_and_eigenvectors(data):
    covariance = get_covariance(data[:, 0:4])
    eigen = la.eig(covariance)
    eigenvalues = eigen[0]
    eigenvectors = np.expand_dims(eigen[1], axis=2)

    for index in range(eigenvectors.shape[0]):
        eigenvectors[index] = eigenvectors[index] / np.linalg.norm(eigenvectors[index])
    
    return eigenvalues, eigenvectors


def pca(data, eigenvectors):
    data_mean = get_data_mean(data[:, :, 0])
    data_number = data.shape[0]
    eigenvector_number = eigenvectors.shape[1]
    eigenvectors = eigenvectors.reshape((eigenvectors.shape[0], eigenvectors.shape[1]))

    output = np.zeros(shape=(eigenvector_number, 1))
    for data_index in range(data_number):
        output = np.hstack((output, eigenvectors.T.dot(data[data_index] - data_mean.T)))
    
    return output[:, 1:]


def get_accuracy(data, label, eigenvectors):
    pca_data = pca(data, eigenvectors)
    accuracy = two_fold_cross_validation(pca_data, label)
    return accuracy


def two_fold_cross_validation(data, label):
    x_train, y_train, x_test, y_test = split_data(data, label)
    accuracy_1 = knn_model(x_train, y_train, x_test, y_test)
    
    x_train, x_test = x_test, x_train
    y_train, y_test = y_test, y_train
    accuracy_2 = knn_model(x_train, y_train, x_test, y_test)

    accuracy = (accuracy_1 + accuracy_2) / 2
    return accuracy


def split_data(data, label):
    x_train = np.concatenate((data[:, 0:25], data[:, 50:75], data[:, 100:125]), axis=1)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1])).T
    
    x_test = np.concatenate((data[:, 25:50], data[:, 75:100], data[:, 125:150]), axis=1)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1])).T

    y_train = np.concatenate((label[0:25], label[50:75], label[100:125]), axis=0)
    y_test = np.concatenate((label[25:50], label[75:100], label[125:150]), axis=0)

    return x_train, y_train, x_test, y_test


def knn_model(x_train, y_train, x_test, y_test):
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(x_train, y_train)

    return knn_model.score(x_test, y_test)

