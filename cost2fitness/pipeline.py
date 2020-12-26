
from collections.abc import Iterable
import numpy as np

from .transformers import BaseTransformer, ReverseByAverage, Min2Zero, Min2Value, AntiMax, AntiMaxPercent, ProbabilityView
from .bar_plots import plot_scores



class Pipeline:

    def __init__(self, transformers):
        
        assert(isinstance(transformers, Iterable)), "transformers should be iterable object (for example, list)"
        assert(len(transformers) > 0), "there should be at least 1 transformer"

        for tf in transformers:
            assert (issubclass(type(tf), BaseTransformer)), "transformer should be subclass of BaseTransformer"

        self.transformers = transformers
    
    def transform(self, array, return_all_steps = False):

        steps = [array]
        for tf in self.transformers:
            steps.append(tf.transform(steps[-1]))
        
        if return_all_steps:
            return np.array(steps)
        return steps[-1]

    def plot_on_example(self, example_array, kind = 'beside', save_as = None):
        
        arr = self.transform(example_array, return_all_steps=True)

        plot_scores(arr, ['at start'] + [tf.name for tf in self.transformers], kind, save_as)




if __name__ == '__main__':


    pipe = Pipeline([
        #ReverseByAverage(),
        #Min2Zero(),
        Min2Value(3),
        #AntiMax(),
        AntiMaxPercent(0.2),
        ProbabilityView()
        ])

    pipe.plot_on_example(np.array([10, 2, 3, 4, 5, 11, 12, 10, 30, 1]), kind = 'under')


