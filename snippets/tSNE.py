'''
Sources
    t-SNE: https://gaussian37.github.io/ml-concept-t_sne/
    Coloring plot by label: https://stackoverflow.com/questions/12487060/color-according-to-class-labels
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

from icecream import ic


def tSNE():

    # load MNIST dataset
    n_classes = 10
    data = load_digits()

    # create t-SNE model
    n_components = 2
    model = TSNE(n_components=n_components)

    # fit embeddings
    embeddings = model.fit_transform(data.data)
    ic(type(data.data), type(embeddings))
    labels = data.target

    # visualize
    if n_components == 2:

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_title('Embedding Space')

        cmap = plt.cm.jet  # mpl.colors.Colormap
        cmaplist = [cmap(i) for i in range(cmap.N)]  # make a list of RGBA tuples
        custom_cmap = cmap.from_list('custom_cmap', cmaplist, cmap.N)

        bounds = np.linspace(0, n_classes, n_classes+1)
        norm = mpl.colors.BoundaryNorm(bounds, custom_cmap.N)

        scat = ax.scatter(embeddings[:,0], embeddings[:,1], c=labels, s=1.5, cmap=custom_cmap, norm=norm)
        cb = plt.colorbar(scat, spacing='proportional',ticks=bounds)

    elif n_components == 3:

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_title('Embedding Space')

        cmap = plt.cm.jet  # mpl.colors.Colormap
        cmaplist = [cmap(i) for i in range(cmap.N)]  # make a list of RGBA tuples
        custom_cmap = cmap.from_list('custom_cmap', cmaplist, cmap.N)

        bounds = np.linspace(0, n_classes, n_classes+1)
        norm = mpl.colors.BoundaryNorm(bounds, custom_cmap.N)

        scat = ax.scatter(embeddings[:,0], embeddings[:,1], embeddings[:,2], c=labels, s=1.5, cmap=custom_cmap, norm=norm)
        cb = plt.colorbar(scat, spacing='proportional',ticks=bounds)

    plt.show()


if __name__ == '__main__':

    tSNE()
