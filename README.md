# lung_seg

3D lung-nodule segmentation with MONAI UNETR.

## Install

```bash
pip install -e .
```

## Data layout

```
data/
├── imagesTr/  # *.nii.gz
└── labelsTr/  # *.nii.gz   (binary masks, same filenames)
```

## Usage

```bash
# single train/val split
python scripts/train.py --config configs/default.yaml

# k-fold cross validation
python scripts/kfold.py --config configs/default.yaml

# inference on a scan
python scripts/predict.py \
    --config configs/default.yaml \
    --model checkpoints/fold0/best.pth \
    --scan /path/to/scan.nii.gz
```