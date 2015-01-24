from __future__ import (absolute_import, division, print_function, unicode_literals)
from supervised_learner import SupervisedLearner
import random

class PerceptronLearner(SupervisedLearner):
    inputs,weights = [],[]
    trainingepoch,unprogressiveEpochs,epochLimit = 0,0,500
    learningrate=.1
    currentAccuracy=0.0

    def train(self, featurematrix, targetmatrix, initialized=False):
      if not initialized:
        self.initInputs(featurematrix,targetmatrix)
      while self.currentAccuracy!=1.0 and self.unprogressiveEpochs!=25 and self.trainingepoch!=self.epochLimit:
        oldaccuracy=self.currentAccuracy
        self.currentAccuracy=runEpoch(targetmatrix)
        self.unprogressiveEpochs=0 if self.currentAccuracy > oldaccuracy else self.unprogressiveEpochs+1
      self.removeBias()
      print("Total epochs run: "+str(self.trainingepoch))

    def runEpoch(self,targetmatrix):
      self.trainingepoch+=1
      for idx,vector in enumerate(self.inputs):
        output=self.checkVector(vector)
        if output!=int(targetmatrix.data[idx][0]):
          self.changeWeights(targetmatrix.data[idx][0],output,vector)
      return self.measure_accuracy(featurematrix,targetmatrix)

    def predict(self, features, targets):
      targets += [self.checkVector(features)]

    def addBias(self,features):
      for inputvector in features:
        inputvector.insert(0,1)
      return features

    def removeBias(self):
      for inputvector in self.inputs:
        inputvector.pop(0)
      self.weights.pop(0)

    def initInputs(self,featurematrix,targetmatrix):
      self.inputs=self.addBias([inputvector for inputvector in featurematrix.data])
      self.weights=[round(random.uniform(-1,1),1) for val in self.inputs[0]]

    def changeWeights(self,target,output,inputvector):
      change=[self.learningrate*(int(target)-output)*inputvector[idx] for idx in range(len(self.weights))]
      self.weights=[round(self.weights[i]+change[i],2) for i in range(len(self.weights))]

    def checkVector(self,vector):
      neuronValue=0
      for k,v in enumerate(vector):
        neuronValue += self.weights[k]*v
      return 1 if neuronValue > 0 else 0

