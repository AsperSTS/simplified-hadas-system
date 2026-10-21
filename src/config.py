import numpy as np
# Load class names (assuming COCO)

TARGET_FPS = 100
FRAME_INTERVAL = 1.0 / TARGET_FPS

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
with open(BASE_DIR / "coco.names", "r") as f:
    CLASS_NAMES = f.read().strip().split("\n")
COLORS_PER_CLASS = np.random.uniform(110, 220, size=(len(CLASS_NAMES), 3))
 