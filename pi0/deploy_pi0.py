import numpy as np
import torch
import hydra
import dill
import sys, os

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
sys.path.append(parent_dir)
from diffusion_policy.workspace.robotworkspace import RobotWorkspace
from diffusion_policy.env_runner.dp_runner import DPRunner

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tyro, os
import sys; sys.path.append('/home/zcj/manipulation/RoboTwin/gello_software')
from gello.env import RobotEnv
from gello.robots.robot import PrintRobot
from gello.zmq_core.robot_node import ZMQClientRobot
from gello.zmq_core.camera_node import ZMQClientCamera
from pynput import keyboard
import torch
from collections import deque
from PIL import Image
import cv2
import time


class DP:

    def __init__(self, ckpt_file: str):
        self.policy = self.get_policy(ckpt_file, None, "cuda:0")
        self.runner = DPRunner(output_dir=None)

    def update_obs(self, observation):
        self.runner.update_obs(observation)

    def get_action(self, observation=None):
        action = self.runner.get_action(self.policy, observation)
        return action

    def get_last_obs(self):
        return self.runner.obs[-1]

    def get_policy(self, checkpoint, output_dir, device):
        # load checkpoint
        payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
        cfg = payload["cfg"]
        print(cfg)
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=output_dir)
        workspace: RobotWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # get policy from workspace
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device(device)
        policy.to(device)
        policy.eval()

        return policy

def encode_obs(observation):
    observation["base_rgb"] = observation["base_rgb"][:,:,[2,1,0]] # RGB to BGR
    head_cam = (np.moveaxis(observation["base_rgb"], -1, 0) / 255).astype(np.float32)
#     front_cam = np.moveaxis(observation['observation']['front_camera']['rgb'], -1, 0) / 255
#     left_cam = (np.moveaxis(observation["observation"]["left_camera"]["rgb"], -1, 0) / 255)
#     right_cam = (np.moveaxis(observation["observation"]["right_camera"]["rgb"], -1, 0) / 255)
    obs = dict(
        head_cam=head_cam,
#         front_cam = front_cam,
#         left_cam=left_cam,
#         right_cam=right_cam,
    )
    obs["agent_pos"] = observation["joint_positions"].astype(np.float32)
    return obs

def get_model(usr_args): # 配置绝对路径
    ckpt_file = f"/home/zcj/manipulation/RoboTwin/policy/DP/checkpoints/{usr_args['task_name']}/{usr_args['checkpoint_num']}.ckpt"
    return DP(ckpt_file)
def reset_model(model):
    model.runner.reset_obs()

def resize_img(image, size=(320,240)):
    # print(image.shape)
    image = Image.fromarray(image)
    image = np.array(image.resize(size, Image.BILINEAR))
    # image = np.transpose(np.array(image), (1,2,0))
    # print(image.shape)
    return image 

@dataclass
class Args:
    agent: str = "none"
    robot_port: int = 6001
    wrist_camera_port: int = 5001
    base_camera_port: int = 5000
    hostname: str = "192.168.1.188" # 主要修改这个
    robot_type: str = None  # only needed for quest agent or spacemouse agent
    hz: int = 100
    start_joints: Optional[Tuple[float, ...]] = None

    gello_port: Optional[str] = None
    mock: bool = False
    use_save_interface: bool = False
    data_dir: str = "/home/landau/gello_software/bc_data"
    task_name: str = 'default' 
    bimanual: bool = False
    verbose: bool = False

def main(args):
    import yaml
    yaml_file = 'deploy_policy.yml'  # 可以是相对路径或绝对路径
    with open(yaml_file, 'r', encoding='utf-8') as file:
        usr_args = yaml.safe_load(file)  # 使用 safe_load 更安全
    policy = get_model(usr_args)

    if args.mock:
        robot_client = PrintRobot(8, dont_print=True)
        camera_clients = {}
    else:
        camera_clients = {
            # you can optionally add camera nodes here for imitation learning purposes
            # "wrist": ZMQClientCamera(port=args.wrist_camera_port, host=args.hostname),
            "base": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),
        }
        robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict=camera_clients)
    count = 0
    reset_model(policy)
    while True: # inference loop
        observation = env.get_obs()
        show_image = cv2.cvtColor(observation['base_rgb'], cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(f'nihao/saved_image{count}.jpg', show_image)
        count+=1
        obs = encode_obs(observation)
        actions = policy.get_action(obs)
        for act in actions:
            act[-1] = 0.12 if act[-1] > 0.4 else 0.7 # 0.12/0.7 are min./max. experimental gripper joint values
            env.step(act)
            observation = env.get_obs()
            obs = encode_obs(observation)
            policy.update_obs(obs)

if __name__ == '__main__':
    main(tyro.cli(Args))