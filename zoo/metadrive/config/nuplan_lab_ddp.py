from easydict import EasyDict
import os
os.environ['SDL_VIDEODRIVER']='dummy'
from metadrive.engine.asset_loader import AssetLoader
from metadrive.policy.replay_policy import ReplayEgoCarPolicy

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
gpu_num = 2
continuous_action_space = True
K = 20  # Number of sampled actions for MCTS.
collector_env_num = 64  # Parallel environments for data collection.
n_episode = int(64*gpu_num)  # Episodes collected per cycle.
evaluator_env_num = 3  # Environments for evaluation.
num_simulations = 50  # MCTS simulations per step.
update_per_collect = 200  # Updates per data collection.
batch_size = 2048  # Samples per training update.
max_env_step = int(1e6)  # Total training steps.

reanalyze_ratio = 0.0  # Ratio of re-evaluated data.
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================
obs_hw=84
obs_shape = [5, obs_hw, obs_hw]
metadrive_sampled_efficientzero_config = dict(
    exp_name=f'data_nuplan/sez_metadrive_old{K}_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed0',
    env=dict(
        env_name='MetaDrive',
        continuous=True,
        obs_shape = obs_shape,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        metadrive=dict(
            log_level=50,
            use_render=False,
            start_scenario_index=0,
            num_scenarios=1800,
            curriculum_level=1,
            distance_penalty=0.01,
            data_directory=AssetLoader.file_path("/zju_0038/pengxiang_workspace/OpenDataLab___nuPlan-v1_dot_1/raw", "metadrive", unix_style=False),
            reactive_traffic=False,
            truncate_as_terminate=False,
            sequential_seed=False,
            crash_vehicle_done=True,
            crash_object_done=True,
            crash_human_done=True,
            out_of_road_penalty=40.0,
            crash_vehicle_penalty=40.0,
            no_negative_reward=True,
            obs_hw=obs_hw,
        ),
    ),
    policy=dict(
        model=dict(
            observation_shape=obs_shape,
            action_space_size=2,
            continuous_action_space=continuous_action_space,
            num_of_sampled_actions=K,
            sigma_type='conditioned',
            model_type='conv',  # options={'mlp', 'conv'}
            lstm_hidden_size=128,
            latent_state_dim=128,
            downsample = True,
            image_channel=5,
        ),
        multi_gpu=True,
        cuda=True,
        env_type='not_board_games',
        game_segment_length=200, # check nuplan dataset
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.0005,
        # NOTE: for continuous gaussian policy, we use the policy_entropy_loss as in the original Sampled MuZero paper.
        policy_entropy_loss_weight=5e-3,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2000),
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
metadrive_sampled_efficientzero_config = EasyDict(metadrive_sampled_efficientzero_config)
main_config = metadrive_sampled_efficientzero_config

metadrive_sampled_efficientzero_create_config = dict(
    env=dict(
        type='metadrive_lightzero',
        import_names=['zoo.metadrive.env.metadrive_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sampled_efficientzero',
        import_names=['lzero.policy.sampled_efficientzero'],
    ),
    collector=dict(
        type='episode_muzero',
        get_train_sample=True,
        import_names=['lzero.worker.muzero_collector'],
    )
)
metadrive_sampled_efficientzero_create_config = EasyDict(metadrive_sampled_efficientzero_create_config)
create_config = metadrive_sampled_efficientzero_create_config
if __name__ == "__main__":
    #  python -m torch.distributed.launch --nproc_per_node=2 ./zoo/metadrive/config/nuplan_lab_ddp.py
    from ding.utils import DDPContext
    from lzero.entry import train_muzero
    from lzero.config.utils import lz_to_ddp_config
    with DDPContext():
        main_config = lz_to_ddp_config(main_config)
        train_muzero([main_config, create_config], seed=1, max_env_step=max_env_step)
