import numpy as np
from dataclasses import dataclass

@dataclass
class Mesh:
    nodes: np.ndarray
    cells: np.ndarray
    cells_centres: np.ndarray
    groups: dict[str,np.ndarray]
    
   