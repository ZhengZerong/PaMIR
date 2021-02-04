#!/usr/bin/python
"""
This script is a wrapper for the training process

You can view the full list of command line options by running `python train.py --help`.
The default values are the ones used to train the models in the paper.
Running the above command will start the training process. It will also create the folders `logs`
and `logs/train_example` that are used to save model checkpoints and Tensorboard logs.
If you start a Tensborboard instance pointing at the directory `logs` you should be able to look
at the logs stored during training.
"""
from util.train_options import TrainOptions
from trainer_geo import Trainer
import torch
import numpy as np

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    torch.manual_seed(31359)
    np.random.seed(31359)
    options = TrainOptions().parse_args()
    trainer = Trainer(options)
    trainer.train()
