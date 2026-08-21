import torch


def get_images():

    front = torch.zeros(
        1,3,480,640
    )


    wrist = torch.zeros(
        1,3,480,640
    )


    return front, wrist
