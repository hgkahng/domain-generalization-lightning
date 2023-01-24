# Domain Generalization Lightning
This repository provides a collection of domain generalization algorithms written in [PyTorch](https://pytorch.org) and [PyTorch Lightning](https://www.pytorchlightning.ai), which is basically a wrapper around pure PyTorch, optimized for research purposes. We encourage readers to go through [the official documentation](https://pytorch-lightning.readthedocs.io/en/stable/) for a high-level understanding of the code structure.

## Installation

```bash
# pip
pip install -r requirements.txt

# conda
conda install --file requirements.txt
```

## Data
```bash
# WILDS
pip install -U wilds
python download_wilds_data.py \
    --root_dir "./data/wilds" \
    --datasets "camelyon17" "poverty" "iwildcam" "rxrx1"

# DomainBed (work in progress)
python download_domainbed_data.py \
    --root_dir "./data/domainbed" \
    --datasets "pacs" "vlcs"
```

## Quick Start
### Empirical Risk Minimization
```bash
# Option 1) Providing command line arguments from the terminal
python dg_lightning/runs/train_erm.py \
    --data "camelyon17" \               # data
    --backbone "densnet121" \           # feature extractor architecture 
    --pretrained \                      # initialize with ImageNet-pretrained weights
    --augmentation \                    # apply basic data augmentations to inputs
    --randaugment \                     # apply RandAugment to inputs
    --optimizer "sgd" \                 # optimizer
    --learning_rate 3e-2 \              # base learning rate
    --weight_decay 1e-5 \               # weight decay factor
    --lr_scheduler "cosine_decay"       # cosine annealing learning rate schedule
    --batch_size 32 \                   # training batch size
    --max_epochs 5 \                    # maximum number of training epochs
    --gpus 0 \                          # use gpu '0'
    --checkpoint_dir "./checkpoints" \  # root checkpoint directory
    --seed 42

# Option 2) Overriding argparse defaults from using a yaml config file (RECOMMENDED)
python dg_lightning/runs/train_erm.py \
    --config dg_lightning/configurations/wilds/camelyon17_train_erm.yaml \  # /path/to/config/file
    --gpus 0 1 2 3 \                                                        # multi-gpu support

# Option 3) additional command line arguments override those specified in the yaml config file
python dg_lightning/runs/train_erm.py \
    --config dg_lightning/configurations/wilds/camelyon17_train_erm.yaml \
    --gpus 0 1 2 3 4 5 6 7 \
    --train_domains 0 1 3 4 \
    --test_domains 2 \
    --optimizer "adam" \
    --max_epochs 30 \
    --seed 2023
```
### HeckmanDG
To reproduce the numbers reported in the paper ([version on OpenReview](https://openreview.net/forum?id=fk7RbGibe1&noteId=Ilmz19EVto7)) on Camelyon17, first train the domain selection model ($\mathbf{g}$):  
```bash
python dg_lightning/runs/train_hdg_selection_model.py \
    --config dg_lightning/configurations/wilds/camelyon17_train_hdg_selection_model.yaml \
    --checkpoint_dir "./checkpoints" \
    --hash "dev0" \
    --gpus 0   # TODO: use device number of your choice
```
which would create a checkpoint file under the folder `./checkpoints/HeckmanDGDomainClassifier/dev0/`. Without the hash argument, a hash will be automatically created based on the current datetime (e.g., 2023-05-01_00:00:00). The checkpoint file will be in the format of `epoch={epoch}-step={step}.ckpt`.

The next step is to train the outcome model ($f$) with the selection model weights trained as stated above. Run the following:
```bash
python dg_lightning/runs/train_hdg.py \
    --config dg_lightning/configurations/wilds/camelyon17_train_hdg.yaml \
    --pretrained_g_ckpt "/path/to/ckpt" \
    --freeze_g_encoder \    # freeze feature extractor
    --freeze_g_predictor \  # freeze linear classifier
    --gpus 0                # TODO: use device number of your choice
```

Alternatively, if you would like to train both $\mathbf{g}$ and ${f}$ simulateously, run the following:
```bash
python dg_lightning/runs/train_hdg.py \
    --config dg_lightning/configurations/wilds/camelyon17_train_hdg.yaml \
    --gpus 0 1 2 3  # We also support multi-gpu distributed training
```

For further details on the argparse arguments:
```bash
# print command line arguments
python dg_lightning/runs/train_hdg_selection_model.py --help
python dg_lightning/runs/train_hdg.py --help
```

## Checklist
- [x] Distributed training on multiple GPUs
- [x] Bash script to download data
- Reproducibility on:
    - [x] Camelyon17
    - [x] PovertyMap
    - [ ] iWildCam
    - [ ] RxRx1
    - [ ] PACS
    - [ ] VLCS
- [x] Learning rate scheduling 
- [ ] A Jupyter notebook tutorial on running HeckmanDG on custom data
