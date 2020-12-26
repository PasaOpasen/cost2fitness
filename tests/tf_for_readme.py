# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 12:15:31 2020

@author: qtckp
"""

import sys
sys.path.append('..')

import numpy as np 

from cost2fitness import Min2Zero

tf = Min2Zero()

arr_of_scores = np.array([10, 8, 7, 5, 8, 9, 20, 12, 6, 18])

tf.transform(arr_of_scores)
# array([ 5,  3,  2,  0,  3,  4, 15,  7,  1, 13])