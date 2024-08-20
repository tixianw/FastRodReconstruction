import numpy as np
from dataclasses import dataclass, field

@dataclass
class ReconstructionResult:
    number_of_elements: int
    position: np.ndarray = field(default=None, init=False)
    director: np.ndarray = field(default=None, init=False)

    def __post_init__(self):
        # Initialize position as a (3, number_of_elements) numpy array
        self.position = np.zeros((3, self.number_of_elements))
        # Initialize director as a (3, 3, number_of_elements) numpy array
        self.director = np.zeros((3, 3, self.number_of_elements+1))
        self.kappa = np.zeros((3, self.number_of_elements-1))
