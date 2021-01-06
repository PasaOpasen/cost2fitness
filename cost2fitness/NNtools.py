
import numpy as np


def mathprod(seq): # math.prod function is available at python 3.8
    """
    returns product of sequence elements
    """
    v = seq[0]
    for s in seq[1:]:
        v *= s
    return v


def arr_to_weigths(arr, shapes):
    """
    Converts 1D-array to list of arrays with needed shapes
    """
    w = []
    k = 0
    for s in shapes:
        count = mathprod(s)
        w.append(arr[k:(k+count)].reshape(s))
        k += count
    return w