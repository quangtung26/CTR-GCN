import os
import numpy as np

class CLSExamplar(object):
    def __init__(self, topo_str, base_dir='cls_matrix'):
        self.A = np.load(os.path.join(os.path.dirname(__file__), base_dir, topo_str + '.npy'))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns


    clsExamplar = CLSExamplar(topo_str="what_will_[J]_act_like_when_[C]-with-punctuation", base_dir='cls_matrix_NWUCLA' )
    new_A = clsExamplar.A.mean(0)
    # print(new_A)

    ax = sns.heatmap(new_A, cmap='Blues')
    plt.show()
