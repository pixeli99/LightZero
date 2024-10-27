# from zoo.metadrive.config.nuplan_lab import main_config, create_config
from zoo.metadrive.config.nuplan_one_scene import main_config, create_config
from lzero.entry import eval_muzero
import numpy as np

if __name__ == "__main__":
    # model_path = "./ckpt/ckpt_best.pth.tar"
    model_path = "/zju_0038/pengxiang_workspace/demo_code/LightZero/data_nuplan/single_scene/ckpt/iteration_50000.pth.tar"
    returns_mean_seeds = []
    returns_seeds = []
    seeds = [111,]
    num_episodes_each_seed = 1
    total_test_episodes = num_episodes_each_seed * len(seeds)
    create_config.env_manager.type = 'base'  # Visualization requires the 'type' to be set as base
    main_config.env.evaluator_env_num = 1  # Visualization requires the 'env_num' to be set as 1
    main_config.env.n_evaluator_episode = total_test_episodes
    main_config.env.replay_path = './video'
    main_config.exp_name = f'lz_result/eval/muzero_eval_ls{main_config.policy.model.latent_state_dim}'

    for seed in seeds:
        """
        - returns_mean (:obj:`float`): The mean return of the evaluation.
        - returns (:obj:`List[float]`): The returns of the evaluation.
        """
        returns_mean, returns = eval_muzero(
            [main_config, create_config],
            seed=seed,
            num_episodes_each_seed=num_episodes_each_seed,
            print_seed_details=False,
            model_path=model_path
        )
        returns_mean_seeds.append(returns_mean)
        returns_seeds.append(returns)

    returns_mean_seeds = np.array(returns_mean_seeds)
    returns_seeds = np.array(returns_seeds)

    # Print evaluation results
    print("=" * 20)
    print(f"We evaluated a total of {len(seeds)} seeds. For each seed, we evaluated {num_episodes_each_seed} episode(s).")
    print(f"For seeds {seeds}, the mean returns are {returns_mean_seeds}, and the returns are {returns_seeds}.")
    print("Across all seeds, the mean reward is:", returns_mean_seeds.mean())
    print("=" * 20)