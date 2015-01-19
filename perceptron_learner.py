from __future__ import (absolute_import, division, print_function, unicode_literals)

from supervised_learner import SupervisedLearner
from matrix import Matrix
import random


class PerceptronLearner(SupervisedLearner):
    inputs=[]
    weights=[]
    targets=[]
    learningrate=0.1
    trainingepoch=0
    epochlimit=15

    def __init__(self):
        pass
      
    def initInputs(self,featurematrix,targetmatrix):
      for inputvector in featurematrix.data:
        inputvector.insert(0,1) # Add bias node
        self.inputs.append(inputvector)
      for val in self.inputs[0]:
        self.weights.append(round(random.uniform(-1,1),1)) # random weights between -1 and 1, rounded to one decimal
      for target in targetmatrix.data:
        self.targets.append(targetmatrix)

    def train(self, featurematrix=None, targetmatrix=None, initialized=False):
        allcorrect=True
        if not initialized:
          self.initInputs(featurematrix,targetmatrix)
        for idx,vector in enumerate(self.inputs):
          output=self.testVector(vector)
          print(self.targets[idx])
          if output!=self.targets[idx]:
            self.changeWeights(self.targets[idx],output,vector)
            allcorrect=False
        if allcorrect or self.trainingepoch == self.epochlimit:
          print (self.targets)
          for inputvector in self.inputs:
            inputvector.pop(0) # remove our bias nodes now that we're done training
        else:
          self.trainingepoch+=1
          self.train(initialized=True)
        
    def test(self, inputvectors, targets):
      pass
        
    def testVector(self,vector):
      curractivation=0
      for k,i in enumerate(vector):
        curractivation += self.weights[k]*i
      if curractivation > 0:
        return 1
      else:
        return 0

    def predict(self, inputvector, targets):
        del targets[:]
        targets += [self.testVector(inputvector)]

    def changeWeights(self,target,output,inputvector):
      for idx,weight in enumerate(self.weights):
        weight+=self.learningrate*(target-output)*inputvector[idx]
      