import numpy as np
import gymnasium as gym
import wandb

from PIL import Image
from icecream import ic


def gif(env_name, agent, fps=30):

    render_mode = 'rgb_array'
    env = gym.make(env_name, render_mode=render_mode)

    frames = []
    o, _ = env.reset()
    frame = env.render()
    frames.append(frame)

    while True:
        with torch.no_grad():
            a, _ = agent.get_action(torch.as_tensor(o, device=agent.device).unsqueeze(0))
            o, _, terminated, truncated, _ = env.step(a.ravel().cpu().numpy())
        d = terminated or truncated
        frame = env.render()
        frames.append(frame)
        if d: break

    frames = [Image.fromarray(frame) for frame in frames]
    frames[0].save('./tmp.gif', save_all=True, append_images=frames[1:], duration=1000/fps, loop=False)


if __name__ == '__main__':

    fps = 30
    env_name = 'BipedalWalker-v3'
    render_mode = 'rgb_array'
    env = gym.make(env_name, render_mode=render_mode)

    frames = []
    o, _ = env.reset()
    frame = env.render()
    frames.append(frame)

    while True:
        a = env.action_space.sample()
        o, _, terminated, truncated, _ = env.step(a)
        d = terminated or truncated
        frame = env.render()
        frames.append(frame)
        if d: break

    frames = [Image.fromarray(frame) for frame in frames]
    frames[0].save('./tmp.gif', save_all=True, append_images=frames[1:], duration=1000/fps, loop=False)

    wandb.init()
    wandb.log({'Episode': wandb.Video('./tmp.gif', format='gif')})
