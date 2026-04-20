import numpy as np
from pathlib import Path


def compute_feature_vector(pair_dir):
    pdb_files = list(Path(pair_dir).glob("*.pdb"))

    if len(pdb_files) == 0:
        return None

    # Dummy Zernike features (replace with your executable pipeline)
    features = []

    for f in pdb_files:
        vec = np.random.rand(50)  # placeholder
        features.extend(vec)

    return np.array(features, dtype=np.float32)
