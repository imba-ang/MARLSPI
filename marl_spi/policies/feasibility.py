import torch


def feasible_actions(lane: int, ramp_merge_available: bool, left_available: bool, right_available: bool):
    if lane not in (1, 2, 3):
        raise ValueError("lane")
    left = (lane == 1 and ramp_merge_available or lane == 2) and left_available
    right = lane == 3 and right_available
    return torch.tensor([True, True, True, left, right], dtype=torch.bool)
