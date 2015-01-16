from __future__ import (absolute_import, division, print_function, unicode_literals)

from toolkitPython.supervised_learner import SupervisedLearner
from toolkitPython.matrix import Matrix


class PerceptronLearner(SupervisedLearner):

    labels = []

    def __init__(self):
        pass

    def train(self, features, labels):

        self.labels = []
        for i in range(labels.cols):
            if labels.value_count(i) == 0:
                self.labels += [0.5]          # continuous
            else:
                self.labels += [0.4]    # nominal

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        del labels[:]
        labels += self.labels



