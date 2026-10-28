from lung_seg.config import Config, load_config


def test_defaults():
    cfg = Config()
    assert cfg.k_folds == 5
    assert cfg.model_name == "unetr"
    assert cfg.roi_size == (96, 96, 96)


def test_load_yaml(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("k_folds: 3\nroi_size: [64, 64, 64]\nmodel_name: unet\n")
    cfg = load_config(p)
    assert cfg.k_folds == 3
    assert cfg.roi_size == (64, 64, 64)
    assert cfg.model_name == "unet"


def test_unknown_keys_ignored(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("k_folds: 4\nnot_a_real_key: 123\n")
    cfg = load_config(p)
    assert cfg.k_folds == 4
    assert not hasattr(cfg, "not_a_real_key")


def test_overrides_win(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("k_folds: 3\nlr: 1.0e-3\n")
    cfg = load_config(p, k_folds=7, lr=2e-4)
    assert cfg.k_folds == 7
    assert cfg.lr == 2e-4