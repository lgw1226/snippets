import torch as th
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    device = th.device('cuda')
    num_hiddens = 10
    model = nn.Sequential(
        nn.Linear(2, num_hiddens),
        nn.ReLU(),
        nn.Linear(num_hiddens, num_hiddens),
        nn.ReLU(),
        nn.Linear(num_hiddens, 1),
    ).to(device)

    num_points = 1000
    range = 10

    x = np.linspace(-range, range, num=num_points, dtype=np.float32)
    y = np.linspace(-range, range, num=num_points, dtype=np.float32)
    x, y = tuple(map(lambda t: th.as_tensor(t, device=device).flatten().unsqueeze(-1), np.meshgrid(x, y)))
    z = model(th.cat((x, y), dim=-1)).squeeze()

    x, y, z = map(lambda t: t.squeeze().reshape(num_points, num_points).cpu().detach(), (x, y, z))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x, y, z, cmap='viridis', linewidth=0, antialiased=False)
    plt.show()
