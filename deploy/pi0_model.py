from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata



def load_pi0(checkpoint):

    dataset = LeRobotDatasetMetadata(
        "piper_pick_all",
        root="/home/thor/thor_vla/dataset/piper_pick_all"
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

    return policy
