import numpy as np
import cvxopt
import cvxopt.solvers


class LinearSVM():
  """
  Linear Support Vector Machine

  Parameters
  ----------
  c: float, default=10.0
      penalty.

  Attributes
  ----------
  weight: ndarray
      A parameter of hyperplane
  
  alpha: ndarray of shape (1, len(x_train))
      A Lagrange multiplier.

  bias: ndarray
      A parameter of hyperplane.
  
  Methods
  -------
  fit(self, x_train, y_train)
      Fit the SVM model according to the given training data.

      x_train: ndarray
          Training data.
      
      y_train: array-like
          Corresponding label of training data.

  evaluate(self, x_test, y_test)
      Return the accuracy on the given test data and labels.

      x_test: ndarray
          Test data.
      
      y_test: array-like
          Corresponding label of test data.

  get_hyperplane_points(self, x_start, x_end)
      Return two corresponding points that y and y' of hyperplane by specifying x_start and x_end.

      x_start: int or float
          x_1.

      x_end: int or float
          x_2.

  Example
  -------
  >>> linear_svm = LinearSVM()
  >>> linear_svm.fit(x_train, y_train)
  >>> print(linear_svm.alpha)
  [[7.90196298e-08]
   [1.19165474e-07]
   ...
   [3.39998461e-08]]
  >>> print(linear_svm.bias)
  [[15.14000151]]
  >>> linear_svm.evaluate(x_test, y_test)
  95.0
  """

  # ↓↓↓ part 1, step 3: training SVM model and solving alpha and bias ↓↓↓
  def __init__(self, c=10.0):
    self.c = c

  def calc_alpha(self):
    x_len = len(self.x_train)
    y_len = len(self.y_train)

    symmetric_matrix = np.zeros(shape=(x_len, x_len))
    for i in range(x_len):
        for j in range(x_len):
            symmetric_matrix[i][j] = self.y_train[i] * self.y_train[j] * self.x_train[i].T.dot(self.x_train[j])

    q = cvxopt.matrix(symmetric_matrix)
    p = cvxopt.matrix(np.ones(x_len) * -1)
    a = cvxopt.matrix(np.array(self.y_train).astype('float'), (1, y_len))
    b = cvxopt.matrix(0.0)

    constrain_1 = np.identity(x_len) * -1
    constrain_2 = np.identity(x_len)
    g = cvxopt.matrix(np.vstack((constrain_1, constrain_2)))
    
    constrain_1 = np.zeros(x_len) * -1
    constrain_2 = np.ones(x_len) * self.c
    h = cvxopt.matrix(np.hstack((constrain_1, constrain_2)))
    
    cvxopt.solvers.options['show_progress'] = False
    self.alpha = cvxopt.solvers.qp(q, p, g, h, a, b)
    self.alpha = np.ravel(self.alpha['x'])
    self.alpha = np.expand_dims(self.alpha, axis=1)

  def calc_weight(self):
    self.weight = 0
    for i in self.support_vector_indices.tolist():
      self.weight += self.y_train[i] * self.alpha[i][0] * self.x_train[i].reshape((2, 1))

  def calc_bias(self):
    first_sv_index = self.support_vector_indices[0]
    self.bias = (1 / self.y_train[first_sv_index]) - self.weight.T.dot(self.x_train[first_sv_index].reshape((2, 1)))

  def calc_support_vector_indices(self):
    self.support_vector_indices = np.where(self.alpha > 10 ** -4)[0]

  def fit(self, x_train, y_train):
    if len(x_train) != len(y_train):
      raise ValueError("Length of training data and label must be same.")
    
    self.x_train = x_train
    self.y_train = y_train

    self.calc_alpha()
    self.calc_support_vector_indices()
    self.calc_weight()
    self.calc_bias()

  # ↑↑↑ part 1, step 3: training SVM model and solving alpha and bias ↑↑↑
  
  # ↓↓↓ part 1, step 4: input test set into decision function of SVM, and get its accuracy ↓↓↓
  def judge_hyperplane_slop(self, w_1, w_2):
      if w_2 == 0: raise ValueError("The weight 2 can't not be zero")
      if w_1 == 0: return 'zero'

      y_1 = (-w_1 * 0 - self.bias) / w_2
      y_2 = (-w_1 * 1 - self.bias) / w_2

      return 'positive' if  y_2 - y_1 > 0 else 'negative'

  def evaluate(self, x_test, y_test):
    w_1, w_2 = self.weight[0], self.weight[1]
    hyperplane_slop = self.judge_hyperplane_slop(w_1, w_2)

    judgement_right_count = 0
    for i in range(len(x_test)):
      data_x = x_test[i, 0]
      data_y = x_test[i, 1]
      hyperplane_y = -(w_1 * data_x + self.bias) / w_2
      
      distance = data_y - hyperplane_y

      if hyperplane_slop == 'positive':
        predict = 1 if distance > 0 else -1
      elif hyperplane_slop == 'negative':
        predict = -1 if distance > 0 else 1 
      else:
        raise ValueError('This function does not support zero slop of hyperplane')

      if predict == 1 and predict == y_test[i]:
        judgement_right_count += 1
      elif predict == -1 and predict == y_test[i]:
        judgement_right_count += 1

    accuracy = (judgement_right_count / len(y_test)) * 100

    return accuracy
  # ↑↑↑ part 1, step 4: input test set into decision function of SVM, and get its accuracy ↑↑↑

  # part 1, step 5: plot the feature and hyperplane on plane of feature 3 - feature 4
  def get_hyperplane_points(self, x_start, x_end):
    w_1, w_2 = self.weight[0], self.weight[1]
    selected_range = range(x_start, x_end + 1, 1)

    x = np.array(selected_range)
    y = list(map(lambda x: (-(w_1 * x + self.bias) / w_2).flatten(), selected_range))
    return x, y


class KernelSVM():
  # ↓↓↓ part 2 & 3, step 3: training SVM model and solving alpha and bias ↓↓↓
  def __init__(self, c=10):
    self.c = c
  
  # mapping() must be override when inherit
  def mapping(self, x_i, x_j):
    return 0

  def calc_alpha(self):
    x_len = len(self.x_train)
    y_len = len(self.y_train)

    symmetric_matrix = np.zeros(shape=(x_len, x_len))
    for i in range(x_len):
        for j in range(x_len):
            symmetric_matrix[i][j] = self.y_train[i] * self.y_train[j] * self.mapping(self.x_train[i].reshape((2, 1)), self.x_train[j].reshape((2, 1)))

    q = cvxopt.matrix(symmetric_matrix)
    p = cvxopt.matrix(np.ones(x_len) * -1)
    a = cvxopt.matrix(np.array(self.y_train).astype('float'), (1, y_len))
    b = cvxopt.matrix(0.0)

    constrain_1 = np.identity(x_len) * -1
    constrain_2 = np.identity(x_len)
    g = cvxopt.matrix(np.vstack((constrain_1, constrain_2)))
    
    constrain_1 = np.zeros(x_len) * -1
    constrain_2 = np.ones(x_len) * self.c
    h = cvxopt.matrix(np.hstack((constrain_1, constrain_2)))
    
    cvxopt.solvers.options['show_progress'] = False
    self.alpha = cvxopt.solvers.qp(q, p, g, h, a, b)
    self.alpha = np.ravel(self.alpha['x'])
    self.alpha = np.expand_dims(self.alpha, axis=1)

  def w_times_x(self, x):
    value = 0
    for i in self.support_vector_indices:
      value += self.y_train[i] * self.alpha[i][0] * self.mapping(self.x_train[i].reshape((2, 1)), x)
    
    return value

  def calc_bias(self):
    first_sv_index = self.support_vector_indices[0]
    self.bias = (1 / self.y_train[first_sv_index]) - self.w_times_x(self.x_train[first_sv_index])

  def calc_support_vector_indices(self):
    self.support_vector_indices = np.where(self.alpha > 10 ** -4)[0]

  def fit(self, x_train, y_train):
    if len(x_train) != len(y_train):
      raise ValueError("Length of training data and label must be same.")
    
    self.x_train = x_train
    self.y_train = y_train

    self.calc_alpha()
    self.calc_support_vector_indices()
    self.calc_bias()

  # ↑↑↑ part 2 & 3, step 3: training SVM model and solving alpha and bias ↑↑↑

  # part 2 & 3, step 4: input test set into decision function of SVM, and get its accuracy
  def evaluate(self, x_test, y_test):
    judgement_right_count = 0
    predictions = self.predict(x_test)

    for i in range(len(x_test)):
      judgement_right_count += 1 if predictions[i] == 1 and y_test[i] == 1 else 0
      judgement_right_count += 1 if predictions[i] == -1 and y_test[i] == -1 else 0

    accuracy = (judgement_right_count / len(y_test)) * 100
    return accuracy

  # ↓↓↓ part 2 & 3, step 5: plot the feature and area of different classes on plane of feature 3 - feature 4 ↓↓↓
  def predict(self, x_test):
    predictions = []
    for i in range(len(x_test)):
      prediction = self.w_times_x(x_test[i]) + self.bias
      prediction = 1 if prediction > 0 else -1
      predictions.append(prediction)
    
    return predictions
  
  def get_binary_class_area(self, start_x=3, end_x=8, start_y=0, end_y=5):
    positive_area = []
    negative_area = []

    for i in np.arange(start_x, end_x, 0.03):
      for j in np.arange(start_y, end_y, 0.03):
        prediction = self.predict(np.array([[i], [j]]).reshape((1, 2, 1)))[0]
        
        if prediction == 1:
          positive_area.append([i, j])
        else:
          negative_area.append([i, j])
    
    positive_area = np.array(positive_area)
    negative_area = np.array(negative_area)
    
    return positive_area, negative_area

  # ↑↑↑ part 2 & 3, step 5: plot the feature and area of different classes on plane of feature 3 - feature 4 ↑↑↑


class RBFSVM(KernelSVM):
  """
  Radial Basis Function-kernel-based (also called Gaussian) Support Vector Machine

  Parameters
  ----------
  c: float, default=10.0
      Penalty.

  sigma: float, default=5.0
      Standard deviation.

  Attributes
  ----------
  weight: ndarray
      A parameter of hyperplane
  
  alpha: ndarray of shape (1, len(x_train))
      A Lagrange multiplier.

  bias: ndarray
      A parameter of hyperplane.
  
  Methods
  -------
  fit(self, x_train, y_train)
      Fit the SVM model according to the given training data.

      x_train: ndarray
          Training data.
      
      y_train: array-like
          Corresponding label of training data.

  evaluate(self, x_test, y_test)
      Return the accuracy on the given test data and labels.
  
      x_test: ndarray
          Test data.
      
      y_test: array-like
          Corresponding label of test data.

  predict(self, x_test)
      Perform classification on samples in x_test.

      x_test: ndarray
          Test data.

  get_binary_class_area(self, start_x=3, end_x=8, start_y=0, end_y=5)
      Return numpy array that area of different classes.

      x_start: int or float
          x_1.

      x_end: int or float
          x_2.

  Example
  -------
  >>> rbf_svm = RBFSVM()
  >>> rbf_svm.fit(x_train, y_train)
  >>> print(rbf_svm.alpha)
  [[7.90196298e-08]
   [1.19165474e-07]
   ...
   [3.39998461e-08]]
  >>> print(rbf_svm.bias)
  [[15.14000151]]
  >>> rbf_svm.evaluate(x_test, y_test)
  95.0
  """

  def __init__(self, c=10.0, sigma=5.0):
    super().__init__(c)
    self.sigma = sigma
  
  def mapping(self, x_i, x_j):
    x_i_minus_x_j = x_i - x_j
    return np.exp((x_i_minus_x_j).T.dot(x_i_minus_x_j) / (-2 * self.sigma ** 2))


class PolynomialSVM(KernelSVM):
  """
  Polynomial-based Support Vector Machine

  Parameters
  ----------
  c: float, default=10.0
      Penalty.

  sigma: float, default=5.0
      Standard deviation.

  Attributes
  ----------
  weight: ndarray
      A parameter of hyperplane
  
  alpha: ndarray of shape (1, len(x_train))
      A Lagrange multiplier.

  bias: ndarray
      A parameter of hyperplane.
  
  Methods
  -------
  fit(self, x_train, y_train)
      Fit the SVM model according to the given training data.

      x_train: ndarray
          Training data.
      
      y_train: array-like
          Corresponding label of training data.

  evaluate(self, x_test, y_test)
      Return the accuracy on the given test data and labels.
  
      x_test: ndarray
          Test data.
      
      y_test: array-like
          Corresponding label of test data.

  get_binary_class_area(self, start_x=3, end_x=8, start_y=0, end_y=5)
      Return numpy array that area of different classes.

      x_start: int or float
          x_1.

      x_end: int or float
          x_2.

  predict(self, x_test)
      Perform classification on samples in x_test.

      x_test: ndarray
          Test data.

  Example
  -------
  >>> polynomial_svm = PolynomialSVM()
  >>> polynomial_svm.fit(x_train, y_train)
  >>> print(polynomial_svm.alpha)
  [[7.90196298e-08]
   [1.19165474e-07]
   ...
   [3.39998461e-08]]
  >>> print(polynomial_svm.bias)
  [[15.14000151]]
  >>> polynomial_svm.evaluate(x_test, y_test)
  95.0
  """

  def __init__(self, c=10, p=2):
    super().__init__(c)
    self.p = p
  
  def mapping(self, x_i, x_j):
    return x_i.T.dot(x_j) ** self.p