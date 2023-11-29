import numpy as np
import matplotlib.pyplot as plt

from icecream import ic


def visualize_2d():

    n_samples = 100

    x = np.random.randn(n_samples)
    y = np.random.randn(n_samples)

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.set_title('Random Data')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')

    ax.plot(x, y, 'k.')

    plt.show()

def visualize_3d():

    n_samples = 100

    x = np.random.randn(n_samples)
    y = np.random.randn(n_samples)
    z = np.random.randn(n_samples)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.set_title('Random Data')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')

    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    # ax.set_zlim(-1, 1)
    ax.set_aspect('equal')

    ax.plot(x, y, z, 'k.')

    plt.show()


if __name__ == '__main__':

    # visualize_2d()
    visualize_3d()
