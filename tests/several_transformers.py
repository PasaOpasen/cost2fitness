import sys
sys.path.append('..')

import numpy as np 

from cost2fitness import plot_scores, ReverseByAverage, AntiMax, AntiMaxPercent, Min2Zero, Min2Value, ProbabilityView, SimplestReverse, AlwaysOnes,NewAvgByMult,NewAvgByShift


transformers = [
    ReverseByAverage(), 
    AntiMax(), 
    AntiMaxPercent(percent = 0.5), 
    Min2Zero(), 
    Min2Value(2), 
    ProbabilityView(), 
    SimplestReverse(), 
    AlwaysOnes(),
    NewAvgByMult(new_average=5),
    NewAvgByShift(new_average=5)
]

plot_beside = [
    True,
    True,
    True,
    True,
    True,
    False,
    False,
    True,
    True
    ]


arr = np.array([10, 9, 8, 7, 6, 5, 5, 5, 8, 9, 12, 15])

# plot

for tf, is_beside in zip(transformers, plot_beside):

    arr2D = np.vstack((arr, tf.transform(arr)))

    plot_scores(arr2D, names = ['start array', tf.name], kind = 'beside' if is_beside else 'under', save_as = f"{tf.name} example.png")








