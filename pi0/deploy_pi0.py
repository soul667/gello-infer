import logging
import os
import pathlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import signal
import sys
import cv2
import jax
import jax.numpy as jnp
import numpy as np
import tyro

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config, data_loader as _data_loader
import openpi.policies.policy as _policy
from gello.env import RobotEnv
from gello.robots.robot import PrintRobot
from gello.zmq_core.camera_node import ZMQClientCamera
from gello.zmq_core.robot_node import ZMQClientRobot

from openpi.policies import rcvlab_policy
import datetime

# 会将observation 以及执行的动作进行录制以便后回放
@dataclass
class VLARecorder:
    """视频录制器，封装所有录制逻辑"""
    video_out_path: str
    frame_size: Tuple[int, int] = (640, 480)  # (width, height)
    fps: int = 30
    fourcc_str: str = 'mp4v'
    camera_key: str = 'base'  # observation中图像的键名
    enabled: bool = True

    writer: Optional[cv2.VideoWriter] = field(default=None, init=False)
    _video_path: Optional[Path] = field(default=None, init=False)

    def __post_init__(self):
        if not self.enabled:
            return
            
        # 创建视频目录
        video_dir = pathlib.Path(self.video_out_path)
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成带时间戳的视频文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._video_path = video_dir / f"episode_{timestamp}.mp4"

        # 创建视频写入器，若失败则尝试 MJPG/AVI 备用方案
        self.writer = self._create_writer(self.fourcc_str, self._video_path)
        if self.writer is None:
            fallback_codec = 'MJPG'
            fallback_path = self._video_path.with_suffix('.avi')
            if self._video_path.exists():
                self._video_path.unlink(missing_ok=True)
            self.writer = self._create_writer(fallback_codec, fallback_path)
            if self.writer is None:
                logging.error("视频写入器初始化失败，禁用录制")
                self.enabled = False
                self._video_path = None
            else:
                logging.warning("mp4v 编码不可用，已切换到 MJPG/AVI")
                self._video_path = fallback_path
                self.fourcc_str = fallback_codec
        if self.writer is not None and self._video_path is not None:
            logging.info(f"视频录制保存到 {self._video_path}，编码 {self.fourcc_str}")
    
    def record_observation(self, observation: dict):
        """从observation中提取并录制图像帧"""
        if not self.enabled or self.writer is None:
            return
            
        if self.camera_key in observation:
            frame = np.asarray(observation[self.camera_key])
            if frame.dtype != np.uint8:
                # 若像素值在 [0, 1] 区间，先放大到 [0, 255]
                max_val = float(frame.max()) if frame.size else 0.0
                if max_val <= 1.0:
                    frame = (frame * 255.0).clip(0, 255)
                frame = frame.clip(0, 255).astype(np.uint8)
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # 如果是 RGB，转换为 BGR (OpenCV 格式)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.write(frame_bgr)
            else:
                self.write(frame)
    
    def write(self, frame):
        """写入一帧"""
        if not self.enabled or self.writer is None:
            return
            
        frame = np.ascontiguousarray(frame)
        if frame.shape[1::-1] != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)
        self.writer.write(frame)
    
    def release(self):
        """释放资源"""
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def _create_writer(self, fourcc_str: str, path: Path) -> Optional[cv2.VideoWriter]:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(path), fourcc, float(self.fps), self.frame_size, True)
        if not writer.isOpened():
            writer.release()
            return None
        return writer
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

@dataclass
class Args:
    # gello connection
    robot_port: int = 6001
    base_camera_port: int = 5000
    hostname: str = "10.27.50.231" # 主要修改这个 为机械臂连接的电脑的IP
    hz: int = 100   # TODO: really 100HZ ?
    mock: bool = False  # TODO: 搞清楚什么意思

    # infer use
    pi_config_name: str = "pi0_rcvlab_low_mem_finetune"   #TODO: change to right config
    checkpoint_dir: str = "/app/openpi/checkpoints/checkpoints/pi0_rcvlab_low_mem_finetune/train1/15000" 
    prompt: str = "place the banana to the box"
    
    # Utils
    video_out_path: str = "/app/videos"  # Path to save videos
    record_video: bool = False  # 是否内录视频


@dataclass
class GelloInferBase:
    args: Args
    env: RobotEnv = None
    recorder: Optional[VLARecorder] = None
            
    def __post_init__(self):
        args = self.args
         # set up robot and cameras
        if args.mock:
            robot_client = PrintRobot(8, dont_print=True)
            camera_clients = {}
        else:
            camera_clients = {
                "base": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),
            }
            logging.info("环境初始化完成1")

            robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
            logging.info("环境初始化完成2")
        
        self.env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict=camera_clients)
        logging.info("环境初始化完成")
        # 获取真实图片尺寸
        frame_size = (640, 480)  # 默认值
        if args.record_video:
            try:
                # 获取一帧图像来确定尺寸
                temp_obs = self.env.get_obs()
                if 'base' in temp_obs:
                    img = temp_obs['base']
                    # frame_size 是 (width, height)
                    frame_size = (img.shape[1], img.shape[0])
                    logging.info(f"检测到图像尺寸: {frame_size}")
            except Exception as e:
                logging.warning(f"无法获取图像尺寸，使用默认值 {frame_size}: {e}")
        
        # 初始化视频录制器（所有逻辑都在 VLARecorder 类中）
        self.recorder = VLARecorder(
            video_out_path=args.video_out_path,
            frame_size=frame_size,
            fps=args.hz,
            camera_key='base',
            enabled=args.record_video
        )
    
    
    def close(self):
        """关闭环境并释放录制器"""
        logging.info("正在关闭环境和录制器...")
        if self.recorder is not None:
            self.recorder.release()
        if self.env is not None:
            # 如果 env 有 close 方法，调用它
            if hasattr(self.env, 'close'):
                self.env.close()
        logging.info("关闭完成")
    
    def __enter__(self):
        """支持 with 语句"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """with 语句结束时自动关闭"""
        self.close()
    
    def __del__(self):
        """对象销毁时确保资源释放"""
        try:
            self.close()
        except:
            pass
@dataclass
class Pi0(GelloInferBase):
    policy: _policy.Policy = None
    config: _config.TrainConfig = None
    key: jnp.ndarray = field(default_factory=lambda: jax.random.PRNGKey(0))

    def __post_init__(self):
        super().__post_init__()
        args = self.args

        self.config = _config.get_config(args.pi_config_name)
        self.policy = _policy_config.create_trained_policy(self.config, args.checkpoint_dir)

    def get_observation(self):
        observation = self.env.get_obs()
        # show observation 的 所有keys
        logging.info(f"Observation keys: {list(observation.keys())}")
        # 录制逻辑完全由 recorder 处理
        self.recorder.record_observation(observation)
        # 转换obs的key
        observation_output = {}
        # joint_positions = np.asarray(observation['joint_positions'], dtype=np.float32)
        # gripper_position = np.asarray(observation['gripper_position'], dtype=np.float32)
        # observation_output['observation/state'] = np.concatenate([joint_positions, gripper_position], axis=None)
        observation_output['observation/state'] = observation['joint_positions'].astype(np.float32)
        # observation_output['observation/image'] = observation['base_rgb'][:,:,[2,1,0]] # BGR to RGB
        observation_output['observation/image'] = observation['base_rgb'][:,:,[2,1,0]] # BGR to RGB
        observation_output['prompt'] = self.args.prompt

        return observation_output
    

@dataclass
class Test(GelloInferBase):
    def __post_init__(self):
        super().__post_init__()
        args = self.args
    def close(self):
        return super().close()

def main(args: Args):    
    # 创建 Pi0 实例
    pi0 = Pi0(args=args)
    # test=Test(args=args)
    
    # 定义信号处理函数
    def signal_handler(sig, frame):
        logging.info("\n收到中断信号 (Ctrl+C)，正在清理资源...")
        pi0.close()
        sys.exit(0)
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        logging.info("开始运行推理循环...")
        # test.get_observation()
        
        while True:
            # print("11111111111")
            obs = pi0.get_observation()
            # print("22222222222")
            result = pi0.policy.infer(obs)
            # 推理动作
            # logging.info(f"推理动作: {result}")
            # print(f"推理动作: {result}")

            for i in range(3, 30):  # 重复执行动作以确保动作被执行
                actions = result['actions'][i, :]
                init_actions = actions.copy()
                # actions[7] = max(0, actions[7] - 0.2)  # 固定夹爪动作
                if actions[7] < 0.63:
                    actions[7] = 0
                else:
                    actions[7] = 1
                # actions[7] = 0.2
                pi0.env.step(actions)
                print(f"执行动作: {actions}  (初始: {init_actions})")

    except KeyboardInterrupt:
        logging.info("\n检测到键盘中断...")
    except Exception as e:
        logging.error(f"发生错误: {e}", exc_info=True)
    finally:
        # 确保无论如何都会关闭
        pi0.close()

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main(tyro.cli(Args))