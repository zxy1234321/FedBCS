# FedBCS

Official implementation of **Divide, Conquer and Unite: Hierarchical Style-Recalibrated Prototype Alignment for Federated Medical Segmentation** (AAAI 2026).

## Overview

FedBCS addresses the domain shift problem in federated medical image segmentation through two core modules:

- **FSR (Frequency-domain Style Recalibration):** Decomposes feature maps into phase and amplitude components via FFT. An adaptive learnable weight blends instance-normalized amplitude with the original, producing style-invariant representations while preserving structural semantics.

- **CDPA (Context-aware Dual-level Prototype Alignment):** Extracts multi-level prototypes from both encoder and decoder paths. Shallow features are fused with deep features through dedicated fusion layers, enabling hierarchical prototype alignment across clients.

Server-side prototype aggregation uses **FINCH clustering** to group semantically similar prototypes before averaging, yielding more representative global prototypes.

### Loss Functions

| Symbol | Description |
|--------|-------------|
| L_dice | CE + Dice joint segmentation loss |
| L_contra | InfoNCE contrastive loss between local features and global prototypes |
| L_consis | MSE consistency regularization toward global prototype centroids |
| L_MP | Multi-level prototype loss = L_contra + L_consis |

## Project Structure

```
FedBCS/
├── main.py                  # Entry point
├── args.py                  # Argument parser & hyperparameters
├── backbone/
│   ├── dac.py               # FSR module (decompose / compose / replace_denormals)
│   └── models.py            # UNet, UNet_FSR, Entropy_Hist
├── fdmodels/
│   ├── fedbcs.py            # FedBCS federated model (core algorithm)
│   └── utils/
│       └── federated_model.py  # Base FederatedModel class
├── fd_trainer/
│   └── training.py          # Training loop & evaluation
├── dataset/
│   ├── myfddataset.py       # Dataset classes (TNBC, MRI)
│   └── utils/               # Dataset utilities, data loading, transforms
├── utils/
│   ├── loss.py              # KDloss, DiceLoss, JointLoss
│   ├── finch.py             # FINCH clustering algorithm
│   ├── best_args.py         # Default hyperparameters per dataset
│   ├── conf.py              # Configuration utilities
│   └── ...
└── config_examples/
    └── training_example.py  # Example configuration
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.10
- MONAI
- torchvision
- numpy
- scipy
- tqdm
- tensorboard
- setproctitle

## Quick Start

### Data Preparation

Organize your data under `./data/` following the structure expected by each dataset class. Refer to `dataset/utils/get_date_from_src.py` for path conventions.

### Training

```bash
python main.py \
    --description "fedbcs_tnbc" \
    --model fedbcs \
    --arch fsr \
    --dataset tnbc \
    --communication_epoch 400 \
    --local_epoch 1 \
    --device_id 0
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `fedbcs` | Federated method |
| `--arch` | `fsr` | Backbone architecture (`fsr` or `unet`) |
| `--dataset` | `tnbc` | Dataset (`tnbc` or `mri`) |
| `--communication_epoch` | `400` | Number of communication rounds |
| `--local_epoch` | `1` | Local training epochs per round |
| `--infoNCET` | `0.005` | Temperature for L_contra |
| `--alp` | `1.0` | Prototype loss weight |
| `--layer_config` | `0,1,2,3` | Encoder layers used for prototype extraction |
| `--fold` | `0` | 5-fold cross-validation fold (0 = disabled) |

### Supported Datasets

| Dataset | Domains | Classes |
|---------|---------|---------|
| TNBC | tcia, crc, kirc, tnbc | 2 |
| MRI | BIDMC, HK, I2CVB, ISBI, ISBI_1.5, UCL | 2 |

### Logging

Training logs are saved to `./logs/<dataset>/` by default. TensorBoard logs and text logs (`log.txt`) are generated automatically. The top-10 models ranked by class-1 Dice score are retained.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zhao2026divide,
  title={Divide, Conquer and Unite: Hierarchical Style-Recalibrated Prototype Alignment for Federated Medical Segmentation},
  author={Zhao, Xingyue and Huang, Wenke and Wang, Xingguang and Zhao, Haoyu and Zhuang, Linghao and Jiang, Anwen and Wan, Guancheng and Ye, Mang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={34},
  pages={28760--28768},
  year={2026}
}
```