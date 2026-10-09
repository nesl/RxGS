import numpy as np
from typing import NamedTuple


# ---- point cloud data structure ----
class BasicPointCloud(NamedTuple):
    points : np.array
