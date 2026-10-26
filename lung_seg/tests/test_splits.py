from lung_seg.config import Config
from lung_seg.data.splits import get_file_lists


def _make_dataset(root, n=10):
    img_dir = root / "imagesTr"
    lbl_dir = root / "labelsTr"
    img_dir.mkdir()
    lbl_dir.mkdir()
    for i in range(n):
        (img_dir / f"case{i:02d}.nii.gz").touch()
        (lbl_dir / f"case{i:02d}.nii.gz").touch()


def test_deterministic_split(tmp_path):
    _make_dataset(tmp_path, n=10)
    cfg = Config(data_dir=str(tmp_path), k_folds=5, fold=0, seed=42)

    train1, val1 = get_file_lists(cfg, fold=0)
    train2, val2 = get_file_lists(cfg, fold=0)

    assert train1 == train2
    assert val1 == val2
    assert len(train1) == 8
    assert len(val1) == 2


def test_folds_are_disjoint(tmp_path):
    _make_dataset(tmp_path, n=10)
    cfg = Config(data_dir=str(tmp_path), k_folds=5, fold=0, seed=42)
    _, val0 = get_file_lists(cfg, fold=0)
    _, val1 = get_file_lists(cfg, fold=1)
    val0_names = {f["image"] for f in val0}
    val1_names = {f["image"] for f in val1}
    assert val0_names.isdisjoint(val1_names)
