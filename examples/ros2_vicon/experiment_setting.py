import numpy as np


class ExpConfig:
    def __init__(self, exp: int):
        self.exp = exp
        self.lab_angle = get_lab_frame_angle(exp)
        self.material_angle = get_material_frame_angle(exp)
        self.material_frame_translation = get_material_frame_translation(exp)

    @classmethod
    def load(cls, exp: int) -> "ExpConfig":
        return cls(exp)


def get_lab_frame_angle(exp: int) -> float:
    if exp == 1:
        angle = -126.0
    elif exp == 2:
        angle = -100.0
    elif exp == 3:
        angle = -100.0
    else:
        raise ValueError(f"Experiment {exp} not supported")
    return angle / 180.0 * np.pi


def get_material_frame_angle(exp: int) -> float:
    if exp == 1:
        angle = -126.0
    elif exp == 2:
        angle = -126.0
    elif exp == 3:
        angle = -100.0
    else:
        raise ValueError(f"Experiment {exp} not supported")
    return angle / 180.0 * np.pi


def get_material_frame_translation(exp: int) -> np.ndarray:
    if exp == 1:
        translation = np.array([-0.009, 0.0055, 0.0])
    elif exp == 2:
        translation = np.array([-0.009, 0.0055, 0.0])
    elif exp == 3:
        translation = np.array([-0.009, 0.0055, 0.0])
    else:
        raise ValueError(f"Experiment {exp} not supported")
    return translation
