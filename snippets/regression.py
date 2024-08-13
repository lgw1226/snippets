import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import matplotlib.pyplot as plt


if __name__ == '__main__':

    device = th.device('cuda')

    model = nn.Sequential(
        nn.Linear(1, 100),
        nn.ReLU(),
        nn.Linear(100, 1),
        nn.ReLU(),
    ).to(device)
    optim = SGD(model.parameters(), lr=0.01)

    num_points = 1000
    x = th.randn((num_points, 1)).to(device)
    y = (x ** 2).to(device)

    num_steps = 5000
    for _ in range(num_steps):
        y_hat = model(x)
        loss = F.mse_loss(y_hat, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(loss)

    for p in model.parameters():
        print(p)

    x_eval = th.linspace(-5, 5, num_points).unsqueeze(-1)
    y_eval = model(x_eval.to(device))

    plt.figure()
    plt.plot(x_eval.cpu().detach(), y_eval.cpu().detach(), color='black')
    plt.scatter(x.cpu().detach(), y.cpu().detach(), marker='.', alpha=0.1, color='red')
    plt.show()