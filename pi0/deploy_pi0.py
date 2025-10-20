import logging
import os
import pathlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import signal
import sys
import cv2
import jax
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
    
    writer: cv2.VideoWriter = None
    _video_path: str = None
    
    def __post_init__(self):
        if not self.enabled:
            return
            
        # 创建视频目录
        video_dir = pathlib.Path(self.video_out_path)
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成带时间戳的视频文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._video_path = str(video_dir / f"episode_{timestamp}.mp4")
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*self.fourcc_str)
        self.writer = cv2.VideoWriter(self._video_path, fourcc, float(self.fps), self.frame_size)
    
    def record_observation(self, observation: dict):
        """从observation中提取并录制图像帧"""
        if not self.enabled or self.writer is None:
            return
            
        if self.camera_key in observation:
            frame = observation[self.camera_key]
            # 如果是 RGB，转换为 BGR (OpenCV 格式)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.write(frame_bgr)
    
    def write(self, frame):
        """写入一帧"""
        if not self.enabled or self.writer is None:
            return
            
        if frame.shape[1::-1] != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)
        self.writer.write(frame)
    
    def release(self):
        """释放资源"""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

@dataclass
class Args:
    # gello connection
    robot_port: int = 6001
    base_camera_port: int = 5000
    hostname: str = "192.168.1.188" # 主要修改这个 为机械臂连接的电脑的IP
    hz: int = 10   # TODO: really 100HZ ?
    mock: bool = False  # TODO: 搞清楚什么意思

    # infer use
    pi_config_name: str = "pi0_default"   #TODO: change to right config
    checkpoint_dir: str = "/home/landau/openpi_checkpoints/" # TODO: change to right checkpoint path
    prompt: str = "pick up the red block"
    
    # Utils
    video_out_path: str = "data/libero/videos"  # Path to save videos
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
            robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
        
        self.env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict=camera_clients)
        
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
        
    def get_observation(self):
        observation = self.env.get_obs()
        # show observation 的 所有keys
        logging.info(f"Observation keys: {list(observation.keys())}")
        # 录制逻辑完全由 recorder 处理
        self.recorder.record_observation(observation)
        return observation
    
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
    key: jax.random.KeyArray = jax.random.key(0)

    def __post_init__(self):
        super().__post_init__()
        args = self.args

        self.config = _config.get_config(args.pi_config_name)
        self.policy = _policy_config.create_trained_policy(self.config, args.checkpoint_dir)

    def infer(self):
        # obs
        pass


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
            obs = pi0.get_observation()  # 自动录制
        #     env.step(outputs['action'])
            
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