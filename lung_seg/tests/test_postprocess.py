import numpy as np

from lung_seg.inference.postprocess import postprocess_prediction


def test_removes_small_keeps_large():
    arr = np.zeros((20, 20, 20), dtype=np.uint8)
    arr[0:2, 0:2, 0:2] = 1          # 8 voxels  → removed
    arr[10:13, 10:13, 10:13] = 1    # 27 voxels → kept
    out = postprocess_prediction(arr, min_size=27)
    assert out[0:2, 0:2, 0:2].sum() == 0
    assert out[11, 11, 11] == 1


def test_empty_input_is_noop():
    arr = np.zeros((8, 8, 8), dtype=np.uint8)
    out = postprocess_prediction(arr, min_size=10)
    assert out.sum() == 0