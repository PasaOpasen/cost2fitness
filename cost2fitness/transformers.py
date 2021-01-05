
from abc import ABC, abstractmethod
import numpy as np

class BaseTransformer(ABC):

    def __init__(self, name):
        
        self.name = name

    @abstractmethod
    def transform(self, array):
        pass



class ReverseByAverage(BaseTransformer):

    def __init__(self):
        self.name = 'Reverse by average'
    
    def transform(self, array):
        
        avg = np.mean(array)*2
        return np.array([(-val + avg) for val in array])

class AntiMax(BaseTransformer):
    
    def __init__(self):
        self.name = 'AntiMax'
    
    def transform(self, array):
        return np.max(array) - array


class AntiMaxPercent(BaseTransformer):

    def __init__(self, percent=0.5):
        assert (percent >= 0 and percent <=1), "percent should be between 0 and 1"

        self.name = f'AntiMax with percent = {percent}'
        self.percent = percent

        self.helper = AntiMax()
    
    def transform(self, array):

        return self.helper.transform(array) + self.percent*np.min(array)

class Min2Zero(BaseTransformer):
    
    def __init__(self):
        self.name = 'Min to zero'
    
    def transform(self, array):
        
        mn = np.min(array)

        return array - mn

class Min2Value(BaseTransformer):
    
    def __init__(self, value):
        self.name = f'Min to value = {value}'
        
        self.value = value
        self.helper = Min2Zero()
    
    def transform(self, array):
        
        return self.helper.transform(array) + self.value


class ProbabilityView(BaseTransformer):
    def __init__(self):
        self.name = 'Prob. view'
    
    def transform(self, array):
        return array/np.sum(array)   

class Softmax(BaseTransformer):
    def __init__(self):
        self.name = 'Softmax'
    
    def transform(self, array):
        arr = np.exp(array)
        return arr/np.sum(arr)   

class Relu(BaseTransformer):
    def __init__(self):
        self.name = 'Relu'
    
    def transform(self, array):
        return np.maximum(0, array) 


class SimplestReverse(BaseTransformer):

    def __init__(self):
        self.name = "Simplest reverse"
    
    def transform(self, array):
        return 1/array

class AlwaysOnes(BaseTransformer):
    
    def __init__(self):
        self.name = "Always ones"
    
    def transform(self, array):
        return np.ones_like(array)


class NewAvgByMult(BaseTransformer):
    
    def __init__(self, new_average):
        self.name = f"Multiple new average = {new_average}"
        self.avg = new_average
    
    def transform(self, array):
        return array * (self.avg/np.mean(array))

class NewAvgByShift(BaseTransformer):
    
    def __init__(self, new_average):
        self.name = f"Shifted new average = {new_average}"
        self.avg = new_average
    
    def transform(self, array):
        return array + (self.avg - np.mean(array))