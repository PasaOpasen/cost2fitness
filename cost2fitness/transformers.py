
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



class Divider(BaseTransformer):

    def __init__(self, divider_number_or_array):
        self.name = "Divider"

        assert (type(divider_number_or_array) in (int, float, np.ndarray)), "divider should be number or 1D array!"

        self.arr = divider_number_or_array
    
    def transform(self, array):
        return array / self.arr


class Argmax(BaseTransformer):

    def __init__(self):
        self.name = "argmax"
    
    def transform(self, array):
        return np.argmax(array)


class Prob2Class(BaseTransformer):

    def __init__(self, threshold = 0.5):
        self.name = "probability to class"
        self.threshold = threshold
    
    def transform(self, array):
        return np.int8(array > self.threshold)


class ProbabilityView(BaseTransformer):
    def __init__(self):
        self.name = 'Prob. view'
    
    def transform(self, array):
        return array/np.sum(array)  


class ToNumber(BaseTransformer):
    def __init__(self):
        self.name = 'to one number'
    
    def transform(self, array):
        return array[0] 


#
# activations
#


class Softmax(BaseTransformer):
    def __init__(self):
        self.name = 'softmax'
    
    def transform(self, array):
        arr = np.exp(array-array.max())
        return arr/np.sum(arr)   

class Relu(BaseTransformer):
    def __init__(self):
        self.name = 'relu'
    
    def transform(self, array):
        return np.maximum(0, array) 

class LeakyRelu(BaseTransformer):
    def __init__(self, alpha = 0.01):
        self.name = 'leaky_relu'
        self.alpha = alpha
    
    def transform(self, array):
        r = arr.copy()
        r[r<0] *= self.alpha
        return r

class Sigmoid(BaseTransformer):
    def __init__(self):
        self.name = 'sigmoid'
    
    def transform(self, array):
        e = np.exp(-array)
        return 1/(1+e)

class Tanh(BaseTransformer):
    def __init__(self):
        self.name = 'tanh'

        #self.sigmoid = Sigmoid()
    
    def transform(self, array):
        return np.tanh(array) #2*self.sigmoid.transform(2*array) - 1 


class ArcTan(BaseTransformer):
    def __init__(self):
        self.name = 'arctan'
    
    def transform(self, array):
        return np.arctan2(array)

class Swish(BaseTransformer):

    def __init__(self, beta = 0.25):
        self.name = 'swish'
        self.beta = beta
        self.sigmoid = Sigmoid()
    
    def transform(self, array):
        return array*self.sigmoid.transform(array*self.beta)

class Softplus(BaseTransformer):
    def __init__(self):
        self.name = 'softplus'
    
    def transform(self, array):
        return np.log(np.exp(array)+1)

class Softsign(BaseTransformer):
    def __init__(self):
        self.name = 'softsign'
    
    def transform(self, array):
        return array / (np.abs(array) + 1)


class Elu(BaseTransformer):
    def __init__(self, alpha = 1.0):
        self.name = 'elu'
        self.alpha = alpha
    
    def transform(self, array):
        answer = array.copy()
        mask = answer < 0
        answer[mask] = self.alpha * ( np.exp(answer[mask]) - 1 )
        return answer



class Selu(BaseTransformer):
    def __init__(self, alpha=1.6732632, scale=1.05070098):
        self.name = 'selu'
        self.scale = scale
        self.elu = Elu(alpha)
    
    def transform(self, array):
        return self.scale * self.elu.transform(array)


#
# Experimental 
#


class Bias(BaseTransformer):
    def __init__(self, bias_len, bias_array = None):
        
        self.name = 'add bias'

        self.get_shape = lambda: [(bias_len,)]

        if bias_array is None:
            self.bias = np.random.uniform(-1, 1, bias_len)
        else:
            assert (bias_len == bias_array.size)
            self.bias = bias_array
    
    def transform(self, array):
        return array + self.bias
    
    def set_weights(self, weights):
        self.bias = weights

class MatrixDot(BaseTransformer):
    def __init__(self, from_size, to_size, matrix_array = None):
        
        self.name = 'matrix multiplication'

        #shape = (to_size, from_size)
        shape = (from_size, to_size)

        self.get_shape = lambda: [shape]

        if matrix_array is None:
            self.matrix = np.random.uniform(-1, 1, shape)
        else:
            assert (shape == matrix_array.shape)
            self.matrix = matrix_array
    
    def transform(self, array):
        #return self.matrix.dot(array[:, np.newaxis]).ravel()
        return array[np.newaxis, :].dot(self.matrix).ravel()
    
    def set_weights(self, weights):
        self.matrix = weights


class NNStep(BaseTransformer):
    def __init__(self, from_size, to_size, matrix_array = None, bias_array = None):
        
        self.name = 'NN step'

        self.Dot = MatrixDot(from_size, to_size, matrix_array)
        self.Add = Bias(to_size, bias_array)

        self.get_shape = lambda: self.Dot.get_shape() + self.Add.get_shape()
    

    def transform(self, array):
        return self.Add.transform(self.Dot.transform(array))

    
    def set_weights(self, weights):
        self.Dot.set_weights(weights[0])
        self.Add.set_weights(weights[1])




