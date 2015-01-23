from __future__ import (absolute_import, division, print_function, unicode_literals)
from supervised_learner import SupervisedLearner
import random

class PerceptronLearner(SupervisedLearner):
    traininginputs, testinputs, weights = [],[],[]
    trainingepoch,unprogressiveEpochs,epochLimit = 0,0,100
    learningrate=.1
    currentAccuracy=0.0
    reserve_training=True
 

    def train(self, featurematrix, targetmatrix, initialized=False):
      if not initialized:
        self.initInputs(featurematrix,targetmatrix)
      for idx,vector in enumerate(self.traininginputs):
        output=self.checkVector(vector)
        if output!=int(targetmatrix.data[idx][0]):
          self.changeWeights(targetmatrix.data[idx][0],output,vector)
      accuracy=self.measure_accuracy(featurematrix,targetmatrix)
      if accuracy!=1.0 and self.unprogressiveEpochs!=25 and self.trainingepoch!=self.epochLimit:
        self.trainingepoch+=1
        self.unprogressiveEpochs=0 if accuracy>self.currentAccuracy else self.unprogressiveEpochs+1
        self.currentAccuracy=accuracy
        self.train(featurematrix,targetmatrix,initialized=True)
      else:
        print("Total epochs run: "+str(self.trainingepoch))

    def predict(self, features, targets):
      targets += [self.checkVector(features)]
      
    def addBias(self,inputvector):
      inputvector.insert(0,1)
      return inputvector
      
    def initInputs(self,featurematrix,targetmatrix):
      featurematrix.shuffle(targetmatrix)
      delimiter=int(len(featurematrix.data)*.7)
      self.traininginputs=[self.addBias(inputvector) for inputvector in featurematrix.data[:delimiter]]
      self.testinputs=[self.addBias(inputvector) for inputvector in featurematrix.data[delimiter:]]
      self.weights=[round(random.uniform(-1,1),1) for val in self.traininginputs[0]]

    def changeWeights(self,target,output,inputvector):
      change=[self.learningrate*(int(target)-output)*inputvector[idx] for idx in range(len(self.weights))]
      self.weights=[round(self.weights[i]+change[i],2) for i in range(len(self.weights))]
        
    def checkVector(self,vector):
      neuronValue=0
      for k,v in enumerate(vector):
        neuronValue += self.weights[k]*v
      return 1 if neuronValue > 0 else 0

