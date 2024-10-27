from metadrive.engine.asset_loader import AssetLoader
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.utils import generate_gif
from IPython.display import Image, clear_output
import cv2

# turn on this to enable 3D render. It only works when you have a screen
threeD_render=False 
# Use the built-in datasets with simulator
nuscenes_data=AssetLoader.file_path("/zju_0038/pengxiang_workspace/OpenDataLab___nuPlan-v1_dot_1/raw", "metadrive", unix_style=False)

env = ScenarioEnv(
    {
        "reactive_traffic": False,
        "use_render": threeD_render,
        "agent_policy": ReplayEgoCarPolicy,
        "data_directory": nuscenes_data,
        "num_scenarios": 1,
        "start_scenario_index": 1067,
    }
)

try:
    scenarios={}
    idx = 0
    for seed in range(1066, 1066+5):
        print("\nSimulate Scenario: {}".format(seed))
        o, _ = env.reset()
        semantic_map = seed == 100
        for i in range(1, 100000):
            o, r, tm, tc, info = env.step([0, 3])
            env.render(mode="top_down", 
                       window=False,
                       screen_record=True,
                       text={"Index": seed,
                             "semantic_map": semantic_map},
                       screen_size=(500, 500),
                       semantic_map=semantic_map) # semantic topdown
            if info["replay_done"]:
                print(i)
                break
        scenarios[idx]=env.top_down_renderer.screen_frames
        idx += 1
finally:
    env.close()

# make gif for three scenarios
frames=[]
min_len=min([len(scenario) for scenario in scenarios.values()])
for i in range(min_len):
    frames.append(cv2.hconcat([scenarios[s][i] for s in range(5)]))
    
clear_output()
generate_gif(frames)
Image(open("demo.gif", "rb").read())