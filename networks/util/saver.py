from __future__ import division
import os
import datetime

import torch
import logging
from tqdm import tqdm

class CheckpointSaver(object):
    """Class that handles saving and loading checkpoints during training."""
    def __init__(self, save_dir, save_steps=1000):
        self.save_dir = os.path.abspath(save_dir)
        self.save_steps = save_steps
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.latest_checkpoint = None
        self.get_latest_checkpoint()
        return

    def exists_checkpoint(self, checkpoint_file=None):
        """Check if a checkpoint exists in the current directory."""
        if checkpoint_file is None:
            return False if self.latest_checkpoint is None else True
        else:
            return os.path.isfile(checkpoint_file)

    @staticmethod
    def save_states_to_pt(checkpoint_filename,
                          models, optimizers, epoch, batch_idx, batch_size, dataset_perm, total_step_count):
        checkpoint = {}
        for model in models:
            checkpoint[model] = models[model].state_dict()
        for optimizer in optimizers:
            checkpoint[optimizer] = optimizers[optimizer].state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['batch_idx'] = batch_idx
        checkpoint['batch_size'] = batch_size
        checkpoint['dataset_perm'] = dataset_perm
        checkpoint['total_step_count'] = total_step_count

        torch.save(checkpoint, checkpoint_filename)
        return

    def save_checkpoint(self, models, optimizers, epoch, batch_idx, batch_size,
                        dataset_perm, total_step_count):
        """Save checkpoint."""
        timestamp = datetime.datetime.now()
        checkpoint_filename = os.path.abspath(
            os.path.join(self.save_dir, timestamp.strftime('%Y_%m_%d_%H_%M_%S') + '.pt'))
        CheckpointSaver.save_states_to_pt(
            checkpoint_filename, models, optimizers, epoch, batch_idx, batch_size,
            dataset_perm, total_step_count)

        logging.info(timestamp.strftime('%Y_%m_%d_%H_%M_%S') +
                   ' | Epoch: %d, Iteration %d' % (epoch, batch_idx))
        logging.info('Saving checkpoint file [' + checkpoint_filename + ']')
        return

    def save_latest(self, models, optimizers, epoch, batch_idx, batch_size,
                    dataset_perm, total_step_count):
        """Save checkpoint."""
        checkpoint_filename = os.path.abspath(os.path.join(self.save_dir, 'latest.pt'))
        CheckpointSaver.save_states_to_pt(
            checkpoint_filename, models, optimizers, epoch, batch_idx, batch_size,
            dataset_perm, total_step_count)
        return

    def load_checkpoint(self, models, optimizers, checkpoint_file=None):
        """Load a checkpoint."""
        if checkpoint_file is None:
            print('Loading latest checkpoint [' + self.latest_checkpoint + ']')
            checkpoint_file = self.latest_checkpoint
        checkpoint = torch.load(checkpoint_file)
        for model in models:
            if model in checkpoint:
                models[model].load_state_dict(checkpoint[model])
        for optimizer in optimizers:
            if optimizer in checkpoint:
                optimizers[optimizer].load_state_dict(checkpoint[optimizer])
        return {'epoch': checkpoint['epoch'],
                'batch_idx': checkpoint['batch_idx'],
                'batch_size': checkpoint['batch_size'],
                'dataset_perm': checkpoint['dataset_perm'],
                'total_step_count': checkpoint['total_step_count']}

    def get_latest_checkpoint(self):
        """Get filename of latest checkpoint if it exists."""
        checkpoint_list = []
        has_latest = False
        for dirpath, dirnames, filenames in os.walk(self.save_dir):
            for filename in filenames:
                if filename == 'latest.pt':
                    has_latest = True
                elif filename.endswith('.pt'):
                    checkpoint_list.append(os.path.abspath(os.path.join(dirpath, filename)))
        checkpoint_list = sorted(checkpoint_list)
        if has_latest:
            checkpoint_list.append(os.path.abspath(os.path.join(dirpath, 'latest.pt')))
        self.latest_checkpoint = None if (len(checkpoint_list) is 0) else checkpoint_list[-1]
        return
