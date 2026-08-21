import torch


class PiperRobot:


    def __init__(self):

        print("Piper initialized")


    def get_joint_state(self):

        # j1~j6 + gripper

        return torch.zeros(
            1,7
        )


    def send_command(self, action):

        print(
            "Send:",
            action
        )
