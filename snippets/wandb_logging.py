'''
Sources
    WandB Docs: https://docs.wandb.ai/guides/track/log/media
    HandWritten Digits Dataset: https://scikit-learn.org/stable/datasets/toy_dataset.html#digits-dataset
'''


import numpy as np
import wandb
import time
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits, load_sample_images
from icecream import ic


def init_wandb():

    timestamp = time.strftime('%y%m%d-%H%M%S')

    wandb.init(
        project='LogPractice',
        name='PracticeRun-'+timestamp
    )

def log_scalar():

    init_wandb()

    scalar = np.random.randn()
    ic(scalar)
    wandb.log({
        'Scalar': scalar
    })

def log_digits():

    init_wandb()

    n_classes = 10
    data = load_digits(n_class=n_classes)

    sample_size = 4
    num_iters = 3

    for _ in range(num_iters):

        idx = np.random.randint(0, len(data), (sample_size,))
        images = data.data[idx].reshape(-1, 8, 8) / 16  # divide by 16 for normalization to (0, 1)
        labels = data.target[idx]
        masks = np.random.rand(*images.shape)
        masked_images = images * masks
        images_columns = np.concatenate((images, masks, masked_images), axis=1)

        wandb.log({
            'Digits': [wandb.Image(image, caption=label) for image, label in zip(images_columns, labels)],    
        })

        plt.imshow(np.concatenate(images_columns, axis=1))
        plt.title(str(labels))
        plt.show()

def log_images():

    init_wandb()

    data = load_sample_images()
    names = ['China', 'Flower']

    num_iters = 3

    for _ in range(num_iters):

        images = np.array(data.images) / 255  # divide by 255 for normalization to (0, 1)
        # images.shape = (2, 427, 640, 3) (batch size, height, width, channels)
        # 1. generate random mask from random sampling
        # 2. expand the last dimension
        # 3. repeat the generated (and expanded) random mask (channels) times
        # masks = np.repeat(np.expand_dims(np.random.rand(*images.shape[:-1]), axis=-1), 3, axis=-1)  # uniform distribution
        masks = np.repeat(np.expand_dims(np.clip(np.random.randn(*images.shape[:-1]) * 0.5 + 0.5, 0, 1), axis=-1), 3, axis=-1)  # normal distribution
        masked_images = images * masks
        columns = np.concatenate((images, masks, masked_images), axis=1)

        wandb.log({
            'Images': [wandb.Image(image, caption=names) for image, names in zip(columns, names)],    
        })

        plt.imshow(np.concatenate(columns, axis=1))
        plt.title(names)
        plt.show()

if __name__ == '__main__':

    # log_scalar()
    # log_digits()
    log_images()
