from __future__ import (absolute_import, division, unicode_literals)

from toolkitPython.supervised_learner import SupervisedLearner
from toolkitPython.baseline_learner import BaselineLearner
from toolkitPython.learning_models import PerceptronLearner
from toolkitPython.matrix import Matrix
import random, time

class MLSystemManager:

    def get_learner(self, model):
        modelmap = {
          "baseline": BaselineLearner(),
          "perceptron": PerceptronLearner(),
          #"neuralnet": NeuralNetLearner(),
          #"decisiontree": DecisionTreeLearner(),
          #"knn": InstanceBasedLearner()
        }
        if model in modelmap:
            return modelmap[model]
        else:
            raise Exception("Unrecognized model: {}".format(model))

    def main(self, args):
        file_name = args["arff"]
        learner_name = args["L"]
        eval_method = args["E"][0]
        eval_parameter = args["E"][1] if len(args["E"]) > 1 else None
        print_confusion_matrix = args["verbose"]
        normalize = args["normalize"]
        random.seed(args["seed"]) # Use a seed for deterministic results, if provided (makes debugging easier)

        restext=[]
        
        # load the model
        learner = self.get_learner(learner_name)

        # load the ARFF file
        data = Matrix()
        data.load_arff(file_name)
        if normalize:
            restext.append("Using normalized data")
            data.normalize()

        # print some stats
        restext.append("\nDataset name: {}\n"
              "Number of instances: {}\n"
              "Number of attributes: {}\n"
              "Learning algorithm: {}\n"
              "Evaluation method: {}\n".format(file_name, data.rows, data.cols, learner_name, eval_method))

        if eval_method == "training":

            restext.append("Calculating accuracy on training set...")

            features = Matrix(data, 0, 0, data.rows, data.cols)
            labels = Matrix(data, 0, data.cols-1, data.rows, 1)
            confusion = Matrix()

            start_time = time.time()
            learner.train(features, labels)
            elapsed_time = time.time() - start_time
            restext.append("Time to train (in seconds): {}".format(elapsed_time))

            accuracy = learner.measure_accuracy(features, labels, confusion)
            restext.append("Training set accuracy: " + str(accuracy))

            if print_confusion_matrix:
                restext.append("\nConfusion matrix: (Row=target value, Col=predicted value")
                restext.append(confusion.display())
                restext.append("")

        elif eval_method == "static":

            restext.append("Calculating accuracy on separate test set...")

            test_data = Matrix(arff=eval_parameter)
            if normalize:
                test_data.normalize()

            restext.append("Test set name: {}".format(eval_parameter))
            restext.append("Number of test instances: {}".format(test_data.rows))
            features = Matrix(data, 0, 0, data.rows, data.cols-1)
            labels = Matrix(data, 0, data.cols-1, data.rows, 1)

            start_time = time.time()
            learner.train(features, labels)
            elapsed_time = time.time() - start_time
            restext.append("Time to train (in seconds): {}".format(elapsed_time))

            train_accuracy = learner.measure_accuracy(features, labels)
            restext.append("Training set accuracy: {}".format(train_accuracy))

            test_features = Matrix(test_data, 0, 0, test_data.rows, test_data.cols-1)
            test_labels = Matrix(test_data, 0, test_data.cols-1, test_data.rows, 1)
            confusion = Matrix()
            test_accuracy = learner.measure_accuracy(test_features, test_labels, confusion)
            restext.append("Test set accuracy: {}".format(test_accuracy))

            if print_confusion_matrix:
                restext.append("\nConfusion matrix: (Row=target value, Col=predicted value")
                restext.append(confusion.display())

        elif eval_method == "random":
            restext.append("Calculating accuracy on a random hold-out set...")
            train_percent = float(eval_parameter)
            if train_percent < 0 or train_percent > 1:
                raise Exception("Percentage for random evaluation must be between 0 and 1")
            restext.append("Percentage used for training: {}".format(train_percent))
            restext.append("Percentage used for testing: {}".format(1 - train_percent))

            data.shuffle()

            train_size = int(train_percent * data.rows)
            train_features = Matrix(data, 0, 0, train_size, data.cols-1)
            train_labels = Matrix(data, 0, data.cols-1, train_size, 1)

            test_features = Matrix(data, train_size, 0, data.rows - train_size, data.cols-1)
            test_labels = Matrix(data, train_size, data.cols-1, data.rows - train_size, 1)

            start_time = time.time()
            learner.train(train_features, train_labels)
            elapsed_time = time.time() - start_time
            restext.append("Time to train (in seconds): {}".format(elapsed_time))

            train_accuracy = learner.measure_accuracy(train_features, train_labels)
            restext.append("Training set accuracy: {}".format(train_accuracy))

            confusion = Matrix()
            test_accuracy = learner.measure_accuracy(test_features, test_labels, confusion)
            restext.append("Test set accuracy: {}".format(test_accuracy))

            if print_confusion_matrix:
                restext.append("\nConfusion matrix: (Row=target value, Col=predicted value")
                restext.append(confusion.display())
                restext.append("")

        elif eval_method == "cross":

            restext.append("Calculating accuracy using cross-validation...")

            folds = int(eval_parameter)
            if folds <= 0:
                raise Exception("Number of folds must be greater than 0")
            restext.append("Number of folds: {}".format(folds))
            reps = 1
            sum_accuracy = 0.0
            elapsed_time = 0.0
            for j in range(reps):
                data.shuffle()
                for i in range(folds):
                    begin = int(i * data.rows / folds)
                    end = int((i + 1) * data.rows / folds)

                    train_features = Matrix(data, 0, 0, begin, data.cols-1)
                    train_labels = Matrix(data, 0, data.cols-1, begin, 1)

                    test_features = Matrix(data, begin, 0, end - begin, data.cols-1)
                    test_labels = Matrix(data, begin, data.cols-1, end - begin, 1)

                    train_features.add(data, end, 0, data.rows - end)
                    train_labels.add(data, end, data.cols-1, data.rows - end)

                    start_time = time.time()
                    learner.train(train_features, train_labels)
                    elapsed_time += time.time() - start_time

                    accuracy = learner.measure_accuracy(test_features, test_labels)
                    sum_accuracy += accuracy
                    restext.append("Rep={}, Fold={}, Accuracy={}".format(j, i, accuracy))

            elapsed_time /= (reps * folds)
            restext.append("Average time to train (in seconds): {}".format(elapsed_time))
            restext.append("Mean accuracy={}".format(sum_accuracy / (reps * folds)))

        else:
            raise Exception("Unrecognized evaluation method '{}'".format(eval_method))
            
        concatenated = "\n".join(restext)
        return concatenated.replace("\n","<br/>")
