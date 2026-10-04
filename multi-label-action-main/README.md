# MSQNet Implementation Details

This directory contains the core model architectures, dataloaders, and training scripts for **MSQNet** (ICCVW 2023).

For complete documentation, see the [main README](../README.md) and the [Project Webpage](https://mondalanindya.github.io/MSQNet/).

## Datasets
- **Animal Kingdom**: Download from [Animal Kingdom](https://sutdcv.github.io/Animal-Kingdom/)
- **Charades**: Download from [AllenAI Charades](https://prior.allenai.org/projects/charades)
- **HMDB51 & Thumos 14**: Download via [MMAction2](https://github.com/open-mmlab/mmaction2)
- **Hockey**: Available through the Hockey dataset repository

## Installation
```bash
pip install -r requirements.txt
```

## Running

### Single GPU Training
```bash
python main.py --dataset animalkingdom --model msqnet --data_dir /path/to/datasets --batch_size 16 --epochs 100 --train True
```

### Multi-GPU Distributed Training
```bash
python dist_main.py --dataset animalkingdom --model msqnet --data_dir /path/to/datasets --batch_size 8 --distributed True
```

### Evaluation
```bash
python main.py --dataset animalkingdom --model msqnet --checkpoint /path/to/checkpoint.pth --train False
```

### Supported Models
- `msqnet` (alias for `timesformerclipinitvideoguide`)
- `timesformerclipinit` (text-only query)
- `timesformer`
- `query2labelclipinit`
- `query2label`
- `convit`
- `videomae`
- `adaptformer`
