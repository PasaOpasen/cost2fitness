
import sys
sys.path.append('..')

import numpy as np 

from cost2fitness import plot_scores, ReverseByAverage, AntiMax, AntiMaxPercent, Min2Zero, Min2Value, ProbabilityView, SimplestReverse, AlwaysOnes, NewAvgByMult,NewAvgByShift, Pl






pipe = Pl([
        Min2Value(3),
        AntiMaxPercent(0.2),
        ProbabilityView()
        ])


arr = np.array([10, 9, 8, 7, 6, 5, 5, 5, 8, 9, 12, 15])

pipe.plot_on_example(arr, kind = 'under', save_as='pipe_example_under.png')


pipe = Pl([
        Min2Value(3),
        AntiMaxPercent(0.2),
        NewAvgByMult(4)
        ])

pipe.plot_on_example(arr, kind = 'beside', save_as='pipe_example_beside.png')

