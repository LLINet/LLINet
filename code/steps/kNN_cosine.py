
  
from numpy import *  
import operator  
import math
import tensorflow as tf
import numpy as np


def createDataSet():  

    group = array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])  
    labels = ['A', 'A', 'B', 'B'] 
    return group, labels 
 
def cosine_distance(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"

    v1_sq =  np.inner(v1,v1)
    v2_sq =  np.inner(v2,v2)
    dis = 1 - np.inner(v1,v2) / math.sqrt(v1_sq * v2_sq)
    return dis
   


def kNNClassify(newInput, dataSet, labels, k): 
    global distance 
    distance = [0]* dataSet.shape[0]
    for i in range(dataSet.shape[0]):
        distance[i] = cosine_distance(newInput, dataSet[i])
  
    sortedDistIndices = argsort(distance)  
  
    classCount = {}  
    for i in range(k):
 
        voteLabel = labels[sortedDistIndices[i]]  
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1  
  
 
    maxCount = 0  
    for key, value in classCount.items():  
        if value > maxCount:  
            maxCount = value  
            maxIndex = key  
  
    return maxIndex
    
