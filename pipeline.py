
import numpy as np

from bar_plots import plot_scores



class Pipeline:

    def __init__(self, transformers):

        assert(len(transformers) > 0), f"there should be at least 1 transformer"

        # check type

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

        plot_scores(arr, 'at start' + [tf.name for tf in self.transformers], kind, save_as)






