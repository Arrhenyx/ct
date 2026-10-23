# lung_seg

3D lung-nodule segmentation with MONAI. Supports **UNETR**, **SwinUNETR**, and
plain **UNet** (selectable via config). Full-volume sliding-window validation
and inference with optional flip-TTA.

## Install

```bash
pip install -e ".[dev]"
```

## Data layout

```
data/
├── imagesTr/   # *.nii.gz
└── labelsTr/   # *.nii.gz   (binary masks, same basenames)
```

## Usage

```bash
# train a single fold
lung-seg train --config configs/default.yaml --fold 0

# k-fold cross validation
lung-seg kfold --config configs/default.yaml

# inference
lung-seg predict \
    --config configs/default.yaml \
    --model_path checkpoints/fold0/best.pth \
    --scan /path/to/case.nii.gz \
    --output_dir predictions --tta
```

The repo also exposes thin wrappers in `scripts/` (`train.py`, `kfold.py`,
`predict.py`) that call the same CLI.
