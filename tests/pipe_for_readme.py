# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 12:09:25 2020

@author: qtckp
"""

import sys
sys.path.append('..')

import numpy as np 

from cost2fitness import ReverseByAverage, AntiMax, Min2Zero, Pl

pipe = Pl([
        Min2Zero(),
        ReverseByAverage(),
        AntiMax()        
        ])


arr_of_scores = np.array([10, 8, 7, 5, 8, 9])


# return each result of pipeline transformation (with input)
pipe.transform(arr_of_scores, return_all_steps= True)
#array([[10.        ,  8.        ,  7.        ,  5.        ,  8.        ,
#         9.        ],
#       [ 5.        ,  3.        ,  2.        ,  0.        ,  3.        ,
#         4.        ],
#       [ 0.66666667,  2.66666667,  3.66666667,  5.66666667,  2.66666667,
#         1.66666667],
#       [ 5.        ,  3.        ,  2.        ,  0.        ,  3.        ,
#         4.        ]])

# return only result of transformation
pipe.transform(arr_of_scores, return_all_steps= False)
#array([5., 3., 2., 0., 3., 4.])
