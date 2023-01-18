# Domain Generalization Lightning
This repository provides a collection of domain generalization algorithms written in [PyTorch](https://pytorch.org) and [PyTorch Lightning](https://www.pytorchlightning.ai), which is basically a wrapper around pure PyTorch, optimized for research purposes. We encourage readers to go through [the official documentation](https://pytorch-lightning.readthedocs.io/en/stable/) for a high-level understanding of the code structure.

## Installation

```bash
# pip
pip install -r requirements.txt

# conda
conda install --file requirements.txt
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

# Option 2) Overriding argparse defaults from using a yaml config file
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
```bash
# print command line arguments
python dg_lightning/runs/train_hdg.py --help
python dg_lightning/runs/train_hdg_selection_model.py --help
python dg_lightning/runs/train_hdg_outcome_model.py --help
```

## Checklist
- [x] Distributed training on multiple GPUs
- [ ] Bash script to download data
- [ ] HeckmanDG for multinomial outcomes
- [ ] A Jupyter notebook tutorial on running HeckmanDG on custom data
- [ ] Learning rate scheduling 
