from __future__ import (absolute_import, division, print_function, unicode_literals)

from supervised_learner import SupervisedLearner
from matrix import Matrix


class PerceptronLearner(SupervisedLearner):
    labels = []

    def __init__(self):
        pass

    def train(self, features, labels):

      self.labels = []
      for i in range(labels.cols):
        if labels.value_count(i) == 0:
          self.labels += [labels.column_mean(i)]          # continuous
        else:
          self.labels += [labels.most_common_value(i)]    # nominal

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        del labels[:]
        labels += self.labels



