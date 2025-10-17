import tyro, os ,cv2 ,jax ,logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from dataclasses import dataclass
from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader



@dataclass
class Args:
    agent: str = "none"
    robot_port: int = 6001
    wrist_camera_port: int = 5001
    base_camera_port: int = 5000

    hostname: str = "192.168.1.188" # 主要修改这个 为机械臂连接的电脑的IP
    robot_type: str = None  # only needed for quest agent or spacemouse agent
    hz: int = 100   # TODO: really 100HZ ?
    start_joints: Optional[Tuple[float, ...]] = None

    gello_port: Optional[str] = None
    mock: bool = False
    use_save_interface: bool = False
    data_dir: str = "/home/landau/gello_software/bc_data"
    task_name: str = 'default' 
    bimanual: bool = False
    verbose: bool = False

    pi_config_name: str = "pi0_default"
    checkpoint_dir: str = "/home/landau/openpi_checkpoints/"
    
    video_save_dir: Optional[str] = None  # 存储推理时内录

def main(args: Args):

    config = _config.get_pi_config(args.pi_config_name)
    policy = _policy_config.create_trained_policy(config, args.checkpoint_dir)
    key = jax.random.key(0)

    # Create a model from the checkpoint.
    model = config.model.load(_model.restore_params(checkpoint_dir / "params"))

    # # We can create fake observations and actions to test the model.
    # obs, act = config.model.fake_obs(), config.model.fake_act()
    while True:
        obs = config.model.fake_obs()
        key, subkey = jax.random.split(key)
        outputs = model.apply({"params": model.params}, obs, rngs={"dropout": subkey})


    pass

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(tyro.cli(Args))