
import numpy as np 
import matplotlib.pyplot as plt 



def plot_scores(scores, names = None, kind = 'beside', save_as = None):
    
    pipes, dim = scores.shape

    X = np.arange(dim)
    step = 0.8/pipes

    xlabels = [str(i+1) for i in range(dim)] if dim * pipes < 100 else ''*dim
    titles = names if not (names is None) else ''*pipes

    if kind == 'beside':
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        for i in range(pipes):
            ax.bar(X + i * step, scores[i],  width = step)

        ax.set_xticks(X + (0.8-step)/2)
        
        ax.set_xticklabels(xlabels)

        if not (names is None):
            ax.legend(labels=titles)

    elif kind == 'under':
        
        fig, axes = plt.subplots(pipes, figsize=(2*pipes, 5))
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        for i in range(pipes):
            axes[i].bar(X, scores[i], color = cycle[i],  width = 0.9)
            axes[i].set_xticks(X)
            axes[i].set_xticklabels(xlabels)
            axes[i].set_title(names[i])
        
        fig.tight_layout()
        
    else:
        raise Exception(f"Unknown kind '{kind}', must be 'beside' or 'under'")
    
    if not (save_as is None):
        plt.savefig(save_as, dpi = 200)
    
    plt.show()


if __name__ == '__main__':
    
    scores = np.random.random((4, 20))
    
    plot_scores(scores, names = ['1', '2', '3', '4'], kind = 'under', save_as = None)



