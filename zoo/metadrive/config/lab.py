#!/usr/bin/env python
"""
This file illustrate how to use top-down renderer to provide observation in form of multiple channels of semantic maps.

We let the target vehicle moving forward directly. You can also try to control the vehicle by yourself. See the config
below for more information.

This script will popup a Pygame window, but that is not the form of the observation. We will also popup a matplotlib
window, which shows the details observation of the top-down pygame renderer.

The detailed implementation of the Pygame renderer is in TopDownMultiChannel Class (a subclass of Observation Class)
at: metadrive/obs/top_down_obs_multi_channel.py

We welcome contributions to propose a better representation of the top-down semantic observation!
"""

import random

import matplotlib.pyplot as plt

from metadrive import TopDownMetaDrive
from metadrive.constants import HELP_MESSAGE
from metadrive.examples.ppo_expert.numpy_expert import expert

from metadrive.engine.asset_loader import AssetLoader
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.utils import generate_gif
from IPython.display import Image, clear_output
import cv2

from zoo.metadrive.env.diy_env import ScenarioEnv
# turn on this to enable 3D render. It only works when you have a screen
threeD_render=False 
# Use the built-in datasets with simulator
nuscenes_data=AssetLoader.file_path(AssetLoader.asset_path, "nuscenes", unix_style=False) 

def draw_multi_channels_top_down_observation(obs, show_time=4):
    num_channels = obs.shape[-1]
    assert num_channels == 5
    channel_names = [
        "Road and navigation", "Ego now and previous pos", "Neighbor at step t", "Neighbor at step t-1",
        "Neighbor at step t-2"
    ]
    fig, axs = plt.subplots(1, num_channels, figsize=(15, 4), dpi=80)
    count = 0

    def close_event():
        plt.close()  # timer calls this function after 3 seconds and closes the window

    timer = fig.canvas.new_timer(
        interval=show_time * 1000
    )  # creating a timer object and setting an interval of 3000 milliseconds
    timer.add_callback(close_event)

    for i, name in enumerate(channel_names):
        count += 1
        ax = axs[i]
        ax.imshow(obs[..., i], cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name)
        # print("Drawing {}-th semantic map!".format(count))
    fig.suptitle("Multi-channels Top-down Observation")
    timer.start()
    plt.savefig("1.jpg")


if __name__ == "__main__":
    print(HELP_MESSAGE)
    env = ScenarioEnv(
    {
        "reactive_traffic": False,
        "use_render": threeD_render,
        "agent_policy": ReplayEgoCarPolicy,
        "data_directory": nuscenes_data,
        "num_scenarios": 1,
    }
)
    try:
        o, _ = env.reset()
        for i in range(1, 100000):
            o, r, tm, tc, info = env.step([1.0, 0.])
            env.render(mode="top_down", text={"Quit": "ESC"}, film_size=(2000, 2000))
            if tm or tc:
                env.reset()
            if i % 1 == 0:
                draw_multi_channels_top_down_observation(o, show_time=0.5)  # show time 4s
                # ret = input("Do you wish to quit? Type any ESC to quite, or press enter to continue")
                # if len(ret) == 0:
                #     continue
                # else:
                #     break
    finally:
        env.close()