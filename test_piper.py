from lerobot_robot_piper import PiperFollowerConfig, PiperFollower


config = PiperFollowerConfig(
    port="can5"
)


robot = PiperFollower(config)


robot.connect(
    calibrate=False
)


print(robot.is_connected)


obs = robot.get_observation()


print(obs)


robot.disconnect()
