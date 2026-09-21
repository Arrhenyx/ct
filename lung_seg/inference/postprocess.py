import cc3d
import numpy as np
from scipy import ndimage


def postprocess_prediction(pred: np.ndarray, min_size: int = 27) -> np.ndarray:
    """Keep only CCs >= min_size voxels, then morphological cleanup."""
    pred = pred.astype(np.uint8)
    cc = cc3d.connected_components(pred, connectivity=26)
    if cc.max() > 0:
        sizes = np.bincount(cc.ravel())
        small = np.where(sizes < min_size)[0]
        small = small[small != 0]
        if small.size:
            pred[np.isin(cc, small)] = 0
    pred = ndimage.binary_closing(pred, np.ones((2, 2, 2))).astype(np.uint8)
    pred = ndimage.binary_opening(pred, np.ones((1, 1, 1))).astype(np.uint8)
    return pred
