from lerobot_robot_piper import PiperFollowerConfig, PiperFollower
from lerobot.cameras.opencv import OpenCVCameraConfig


camera_config = {

    "front": OpenCVCameraConfig(
        index_or_path="/dev/video2",
        width=480,
        height=270,
        fps=30,
    ),

    "wrist": OpenCVCameraConfig(
        index_or_path="/dev/video8",
        width=480,
        height=270,
        fps=30,
    ),
}


robot_cfg = PiperFollowerConfig(
    port="can5",
    cameras=camera_config,
)


robot = PiperFollower(robot_cfg)

robot.connect(
    calibrate=False
)


print("connected:", robot.is_connected)


obs = robot.get_observation()


for k,v in obs.items():

    if hasattr(v,"shape"):
        print(k,v.shape)

    else:
        print(k,v)


robot.disconnect()