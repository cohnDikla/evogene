from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
import numpy as np


class Classifier:
    """
    Linear Support Vector Classifier (linear kernel SVM).
    """

    def __init__(self, train_data, train_labels, test_data, test_labels):
        """
        Constructor of the Classifier Class.
        """
        self._train_data = np.reshape(train_data, (len(train_data), 1))
        self._train_labels = train_labels
        self._test_data = np.reshape(test_data, (len(test_data), 1))
        self._test_labels = test_labels
        self._model = None

    def train_classifier(self):
        """
        Train the linear support vector classifier.
        """
        self._model = LinearSVC()
        self._model.fit(self._train_data, self._train_labels)

    def evaluate_classifier(self):
        """
        Evaluates the classifier using the test data.
        :return: the F1 score -  a measure of a test's accuracy,
        which considers both the precision and the recall of the test
        to compute the score.
        """
        predictions = self._model.predict(self._test_data)
        return f1_score(self._test_labels, predictions)
