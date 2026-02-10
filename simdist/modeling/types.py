from typing import TypedDict


class Stats(TypedDict):
    mean: float
    std: float


class ScalerParams(TypedDict):
    proprio_obs: Stats
    extero_obs: Stats
    actions: Stats
    commands: Stats
    rewards: Stats
    values: Stats
