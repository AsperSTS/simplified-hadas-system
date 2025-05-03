import numpy as np
# Load class names (assuming COCO)
CLASS_NAMES = None
with open("coco.names", "r") as f:
    CLASS_NAMES = f.read().strip().split("\n")
COLORS_PER_CLASS = np.random.uniform(85, 190, size=(len(CLASS_NAMES), 3))
