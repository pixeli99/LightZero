from typing import Any, Dict, Optional
from easydict import EasyDict
import matplotlib.pyplot as plt
import gymnasium as gym
import copy
import numpy as np
from ding.envs.env.base_env import BaseEnv, BaseEnvTimestep
from ding.torch_utils.data_helper import to_ndarray
from ding.utils.default_helper import deep_merge_dicts
from ding.utils import ENV_REGISTRY

from zoo.metadrive.env.drive_env import MetaDrive
from zoo.metadrive.env.nuplan_env import ScenarioEnv

import pygame
import os
from PIL import Image
from datetime import datetime

def obs2rgb(obs):
    """
    Overview:
        Combine a multi-channel top-down observation into a single RGB image.
    Auguments:
    - obs (:obj:`numpy.ndarray`): A 3D NumPy array of shape (height, width, 5) representing the observation data,
                    where the last dimension corresponds to the five distinct channels.
    Returns:
    - rgb_image (:obj:`numpy.ndarray`): A 3D NumPy array of shape (height, width, 3) representing the RGB image.
    """
    # Validate that there are exactly five channels in the observation data.
    num_channels = obs.shape[-1]
    assert num_channels == 5, "The observation data must have exactly 5 channels."
    
    # Normalize each channel to be within the range [0, 1]
    obs_normalized = np.clip(obs, 0, 1)
    
    # Define weights for the RGB channels (these can vary depending on the desired styling)
    # Each channel is mapped to a unique color in the resulting RGB space
    r_weight = np.array([1, 0, 0, 0.5, 0])  # Red influence from Channels 0 and 3
    g_weight = np.array([0, 1, 0, 0.5, 0])  # Green influence from Channels 1 and 3
    b_weight = np.array([0, 0, 1, 0, 1])    # Blue influence from Channels 2 and 4
    
    # Create the RGB image by combining the weighted channels
    r = np.sum(obs_normalized * r_weight, axis=-1)
    g = np.sum(obs_normalized * g_weight, axis=-1)
    b = np.sum(obs_normalized * b_weight, axis=-1)
    
    # Stack the arrays along the last axis to form an RGB image
    rgb_image = np.stack([r, g, b], axis=-1)
    
    # Clip the resulting image to ensure values are within [0, 1]
    rgb_image = np.clip(rgb_image, 0, 1)
    
    return rgb_image

def draw_multi_channels_top_down_observation(obs, show_time=0.5):
    """
    Overview:
        Displays a multi-channel top-down observation from an autonomous vehicle.
    Auguments:
    - obs (:obj:`numpy.ndarray`): A 3D NumPy array of shape (height, width, 5) representing the observation data,
                    where the last dimension corresponds to the five distinct channels.
    - show_time (:obj:`float`): The duration in seconds for which the observation image will be displayed. Defaults to 0.5 seconds.
    """
    # Validate that there are exactly five channels in the observation data.
    num_channels = obs.shape[-1]
    assert num_channels == 5, "The observation data must have exactly 5 channels."
    
    # Define the names for each of the five channels.
    channel_names = [
        "Road and navigation", "Ego now and previous pos", "Neighbor at step t", "Neighbor at step t-1",
        "Neighbor at step t-2"
    ]
    
    # Create a figure with a subplot for each channel.
    fig, axs = plt.subplots(1, num_channels, figsize=(15, 4), dpi=80)
    
    # Initialize a counter to track the current channel index.
    count = 0
    
    # Define a callback function to close the figure after the specified show_time.
    def close_event():
        plt.close(fig)  # Explicitly close the figure referenced by 'fig'.
    
    # Create a timer that triggers the close_event after the specified duration.
    timer = fig.canvas.new_timer(interval=show_time * 1000)
    timer.add_callback(close_event)
    
    # Iterate over each channel and display its observation data.
    for i, name in enumerate(channel_names):
        count += 1
        ax = axs[i]  # Retrieve the subplot for the current channel.
        ax.imshow(obs[..., i], cmap="bone")  # Display the observation data using a bone colormap.
        ax.set_xticks([])  # Hide the x-axis ticks.
        ax.set_yticks([])  # Hide the y-axis ticks.
        ax.set_title(name)  # Set the title for the subplot based on the channel name.
    
    # Set a title for the entire figure that summarizes the content.
    fig.suptitle("Multi-channels Top-down Observation", fontsize='large')

    # Start the timer to initiate the automatic closing of the figure.
    timer.start()
    
    plt.savefig('1.jpg')
    # Display the figure with the multi-channel observation data.
    # plt.show()
    
    # Close the figure after it has been displayed for the specified duration.
    # plt.close()  # Explicitly close the figure to ensure it is properly closed.

@ENV_REGISTRY.register('metadrive_lightzero')
class MetaDriveEnv(BaseEnv):
    """
    Overview:
        MetaDrive environment in LightZero.
    """
    config = dict(
        # (bool) Whether to use continuous action space
        continuous=True,
        # replay_path (str or None): The path to save the replay video. If None, the replay will not be saved.
        # Only effective when env_manager.type is 'base'.
        replay_path=None,
        # (bool) Whether to scale action into [-2, 2]
        act_scale=True,

    )
    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg
    
    def __init__(self, cfg: dict = {}) -> None:
        """
        Overview:
            Initialize the environment with a configuration dictionary. Set up spaces for observations, actions, and rewards.
        Arguments:
            - cfg (:obj:`dict`): Configuration dict.
        """
        # Initialize a raw env
        self._cfg = cfg
        # self._env = MetaDrive(self._cfg)
        self._env = ScenarioEnv(self._cfg)
        self._init_flag = True
        self._reward_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(1, ))
        self._action_space = self._env.action_space
        self._observation_space = self._env.observation_space

        # bird view
        self.show_bird_view = True
        self.frames = []
        self.obs_rec = []

    def reset(self, *args, **kwargs) -> Any:
        """
        Overview:
            Reset the environment and return the initial observation.
        Returns:
            - metadrive_obs (:obj:`dict`): An observation dict for the MetaDrive env which includes ``observation``, ``action_mask``, ``to_play``.
        """
        if len(self.frames) > 1 and self.show_bird_view:
            # 获取当前目录下的所有GIF文件
            gif_files = [f for f in os.listdir('./demo_gifs') if f.endswith('.gif')]
            
            # 如果GIF文件数量超过5个，删除最早的文件
            if len(gif_files) >= 5:
                # 按文件的修改时间排序，找到最早的文件
                try:
                    gif_files.sort(key=lambda x: os.path.getmtime(os.path.join('./demo_gifs', x)))
                    os.remove(os.path.join('./demo_gifs', gif_files[0]))
                except:
                    pass
                print(f"Deleted oldest GIF: {gif_files[0]}")
            
            combined_frames = []
            for cur_obs, frame in zip(self.obs_rec, self.frames):
                # 将obs和frame的像素值从[0,1]转换到[0,255]并转换为uint8
                cur_obs_uint8 = (cur_obs * 255).astype(np.uint8)
                frame_uint8 = frame.astype(np.uint8)
                
                # 使用PIL将NumPy数组转换为Image对象
                img_obs = Image.fromarray(cur_obs_uint8)
                img_frame = Image.fromarray(frame_uint8)
                
                # img_obs_resized = img_obs.resize((512, 512))
                # img_frame_resized = img_frame.resize((512, 512))
                
                # 将调整大小后的Image对象转换回NumPy数组
                # arr_obs_resized = np.array(img_obs_resized)
                # arr_frame_resized = np.array(img_frame_resized)
                
                # 拼接图像，沿宽度方向（axis=1）
                # combined = np.concatenate((arr_obs_resized, arr_frame_resized), axis=1)
                
                combined = np.array(img_frame)
                combined_frames.append(combined)
            
            # 将合并后的帧转换为PIL Image格式
            imgs = [Image.fromarray(img) for img in combined_frames]
            
            # 生成保存的文件名
            filename = datetime.now().strftime("demo_%Y%m%d_%H%M%S.gif")
            
            # 保存GIF文件
            imgs[0].save(os.path.join('./demo_gifs', filename), save_all=True, append_images=imgs[1:], duration=50, loop=0)
            print(f"Gif saved as {filename}.")

        
        obs = self._env.reset(*args, **kwargs)
        obs = to_ndarray(obs, dtype=np.float32)
        if isinstance(obs, np.ndarray) and len(obs.shape) == 3:
            obs = obs.transpose((2, 0, 1))
            # obs = obs
        elif isinstance(obs, dict):
            vehicle_state = obs['vehicle_state']
            # birdview = obs['birdview'].transpose((2, 0, 1))
            birdview = obs['birdview']
            obs = {'vehicle_state': vehicle_state, 'birdview': birdview}
        self._eval_episode_return = 0.0
        self._arrive_dest = False
        self._observation_space = self._env.observation_space
        
        metadrive_obs = {}
        metadrive_obs['observation'] = obs 
        metadrive_obs['action_mask'] = None 
        metadrive_obs['to_play'] = -1 
        
        self.frames = []
        self.obs_rec = []
        return metadrive_obs
    
    def step(self, action: np.ndarray = None) -> BaseEnvTimestep:
        """
        Overview:
            Wrapper of ``step`` method in env. This aims to convert the returns of ``gym.Env`` step method into
            that of ``ding.envs.BaseEnv``, from ``(obs, reward, done, info)`` tuple to a ``BaseEnvTimestep``
            namedtuple defined in DI-engine. It will also convert actions, observations and reward into
            ``np.ndarray``. In origin MetaDrive setting the action can be None, but in our pipeline an action is always performed to the environment.
        Arguments:
            - action (:obj:`np.ndarray`): The action to be performed in the environment. 
        Returns:
            - timestep (:obj:`BaseEnvTimestep`): An object containing the new observation, reward, done flag,
              and info dictionary.
        """
        action = to_ndarray(action)
        obs, rew, done, info = self._env.step(action)
        done = (done or info["replay_done"])  # replay_done is a flag to indicate the end of the episode
        frame = self._env.render(mode="top_down",
                                 window=False,
                                 target_vehicle_heading_up=False,
                                 screen_size=(500, 500))
        self.frames.append(frame)
        self.obs_rec.append(obs2rgb(obs))
        if self.show_bird_view:
            draw_multi_channels_top_down_observation(obs, show_time=0.5)
        self._eval_episode_return += rew
        obs = to_ndarray(obs, dtype=np.float32)
        if isinstance(obs, np.ndarray) and len(obs.shape) == 3:
            obs = obs.transpose((2, 0, 1))
            # obs = obs
        elif isinstance(obs, dict):
            vehicle_state = obs['vehicle_state']
            # birdview = obs['birdview'].transpose((2, 0, 1))
            birdview = obs['birdview']
            obs = {'vehicle_state': vehicle_state, 'birdview': birdview}
        rew = to_ndarray([rew], dtype=np.float32)
        if done:
            info['eval_episode_return'] = self._eval_episode_return
        metadrive_obs = {}
        metadrive_obs['observation'] = obs  
        metadrive_obs['action_mask'] = None 
        metadrive_obs['to_play'] = -1 
        return BaseEnvTimestep(metadrive_obs, rew, done, info)
    
    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        """
        Set the seed for the environment's random number generator. Can handle both static and dynamic seeding.
        """
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path
        self._env = gym.wrappers.Monitor(self._env, self._replay_path, video_callable=lambda episode_id: True, force=True)

    def render(self):
        self._env.render()

    @property
    def observation_space(self) -> gym.spaces.Space:
        """
        Property to access the observation space of the environment.
        """
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        """
        Property to access the action space of the environment.
        """
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        """
        Property to access the reward space of the environment.
        """
        return self._reward_space
    
    def close(self) -> None:
        """
        Close the environment, and set the initialization flag to False.
        """
        if self._init_flag:
            self._env.close()
        self._init_flag = False
 
    def __repr__(self) -> str:
        return repr(self._env)

    def clone(self):
        cfg = copy.deepcopy(self._cfg)
        return MetaDriveEnv(cfg)