import torch
import argparse

from transformers import AutoTokenizer

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from lerobot_robot_piper.piper_follower import PiperFollower
from lerobot_robot_piper.config_piper import PiperFollowerConfig



# =====================
# Argument
# =====================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--instruction",
    type=str,
    default="pick up the object"
)

args = parser.parse_args()



# =====================
# Checkpoint
# =====================

checkpoint = (
    "/home/sejong/lerobot_teleop/"
    "model/pretrained_model"
)



# =====================
# Load Pi0
# =====================

dataset = LeRobotDatasetMetadata(
    "piper_pick_all",
    root="/home/sejong/lerobot_teleop/dataset/piper_pick_all"
)


cfg = PreTrainedConfig.from_pretrained(
    checkpoint
)

cfg.pretrained_path = checkpoint


policy = make_policy(
    cfg,
    ds_meta=dataset
)


policy.eval()

print("Pi0 loaded")



# =====================
# Language
# =====================

tokenizer = AutoTokenizer.from_pretrained(
    checkpoint + "/tokenizer"
)


tokens = tokenizer(
    args.instruction,
    padding="max_length",
    max_length=48,
    truncation=True,
    return_tensors="pt"
)


lang_tokens = tokens["input_ids"].cuda()

lang_mask = (
    tokens["attention_mask"]
    .bool()
    .cuda()
)



# =====================
# Piper
# =====================


robot_cfg = PiperFollowerConfig(
    port="/dev/ttyUSB0",
    cameras={}
)


robot = PiperFollower(robot_cfg)

robot.connect(
    calibrate=False
)


print(robot)



# =====================
# Control loop
# =====================


while True:


    obs_robot = robot.get_observation()


    # 여기서 LeRobot obs 변환 필요

    obs = {

    }



    with torch.no_grad():

        action = policy.select_action(obs)



    print(action)


    robot.send_action(action)
