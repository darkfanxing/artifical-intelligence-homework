import numpy as np
import numpy as np
import seaborn as sns
from svm import LinearSVM, RBFSVM, PolynomialSVM


def set_positive_and_negative_data(data, selected_features, split_index=50):
    positive_data = data[:split_index, selected_features]
    positive_data = np.expand_dims(positive_data, axis=2)
    
    negative_data = data[split_index:, selected_features]
    negative_data = np.expand_dims(negative_data, axis=2)

    return positive_data, negative_data


def get_training_and_test_data(positive_data, negative_data, split_index=25):
    train_positive, train_negative = positive_data[:split_index], negative_data[:split_index]
    test_positive, test_negative = positive_data[split_index:], negative_data[split_index:]

    x_train = np.concatenate((train_positive, train_negative), axis=0)
    x_test = np.concatenate((test_positive, test_negative), axis=0)

    y_train = [1] * len(train_positive) + [-1] * len(train_negative)
    y_test = [1] * len(test_positive) + [-1] * len(test_negative)

    y_train_for_plot = ['positive'] * len(train_positive) + ['negative'] * len(train_negative)
    return x_train, x_test, y_train, y_test, y_train_for_plot

# part 1 & 2 & 3, step 6: cross validation
def cross_validation(svm_type, x_train, x_test, y_train, y_test, c=10, sigma=None, p=None):
    accuracys = []
    for count in range(2):
        if svm_type == 'linear_svm':
            svm = LinearSVM(c=c)
            svm.fit(x_train, y_train)
        elif svm_type == 'rbf_svm':
            if not sigma:
                raise ValueError('sigma must have value.')

            svm = RBFSVM(c=c, sigma=sigma)
            svm.fit(x_train, y_train)
        elif svm_type == 'polynomial_svm':
            if not p:
                raise ValueError('p must be number.')

            svm = PolynomialSVM(c=c, p=p)
            svm.fit(x_train, y_train)
        else:
            raise ValueError('The svm_type Parameter must be "linear_svm", "rbf_svm" or "poly_svm".')

        accuracys.append(svm.evaluate(x_test, y_test))
        
        if count == 1:
            break
        else:
            x_train, x_test = x_test, x_train
            y_train, y_test = y_test, y_train
            
    # part 1 & 2 & 3, step 7: get average accuracy
    return np.array(accuracys).mean()


def plot_binary_class_area_and_2d_data(svm, x, hue_label, x_label='Feature 3', y_label='Feature 4'):
    positive_area, negative_area = svm.get_binary_class_area()

    sns.scatterplot(x=positive_area[:, 0], y=positive_area[:, 1])
    sns.scatterplot(x=negative_area[:, 0], y=negative_area[:, 1])
    sns.scatterplot(x=x[:, 0, 0], y=x[:, 1, 0], hue=hue_label).set(xlabel=x_label, ylabel=y_label)