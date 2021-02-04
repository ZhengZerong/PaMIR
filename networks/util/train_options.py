import os
import json
import ast
import argparse
import numpy as np
from collections import namedtuple
from datetime import datetime

from .util import create_code_snapshot


class TrainOptions(object):
    """Object that handles command line options."""
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        req = self.parser.add_argument_group('Required')
        req.add_argument('--name', required=True, help='Name of the experiment')

        gen = self.parser.add_argument_group('General')
        gen.add_argument('--time_to_run', type=int, default=np.inf, help='Total time to run in seconds. Used for training in environments with timing constraints')
        gen.add_argument('--resume', dest='resume', default=False, action='store_true', help='Resume from checkpoint (Use latest checkpoint by default')
        gen.add_argument('--num_workers', type=int, default=8, help='Number of processes used for data loading')
        pin = gen.add_mutually_exclusive_group()
        pin.add_argument('--pin_memory', dest='pin_memory', action='store_true', help='Number of processes used for data loading')
        pin.add_argument('--no_pin_memory', dest='pin_memory', action='store_false', help='Number of processes used for data loading')
        gen.set_defaults(pin_memory=True)

        io = self.parser.add_argument_group('io')
        io.add_argument('--log_dir', default='logs', help='Directory to store logs')
        io.add_argument('--checkpoint', default=None, help='Path to checkpoint')
        io.add_argument('--from_json', default=None, help='Load options from json file instead of the command line')
        io.add_argument('--pretrained_checkpoint', default=None, help='Load a pretrained network when starting training')
        io.add_argument('--pretrained_gcmr_checkpoint', default=None, help='Load a pretrained gcmr network when starting training')
        io.add_argument('--pretrained_pamir_net_checkpoint', default=None, help='Load a pretrained pamir when starting training')

        dataloading = self.parser.add_argument_group('Data Loading')
        dataloading.add_argument('--dataset_dir', type=str, help='dataset directory')
        dataloading.add_argument('--view_num_per_item', type=int, default=60, help='view_num_per_item')
        dataloading.add_argument('--point_num', type=int, default=5000, help='number of point samples')

        train = self.parser.add_argument_group('Training Options')
        train.add_argument('--num_epochs', type=int, default=50, help='Total number of training epochs')
        train.add_argument('--batch_size', type=int, default=16, help='Batch size')
        train.add_argument('--summary_steps', type=int, default=100, help='Summary saving frequency')
        train.add_argument('--checkpoint_steps', type=int, default=20000, help='Checkpoint saving frequency')
        train.add_argument('--test_steps', type=int, default=10000, help='Testing frequency')
        train.add_argument('--use_adaptive_geo_loss', type=ast.literal_eval, dest='use_adaptive_geo_loss', default=False, help='use_adaptive_geo_loss')
        train.add_argument('--use_multistage_loss', type=ast.literal_eval, dest='use_multistage_loss', default=True, help='use_multistage_loss')
        train.add_argument('--use_gt_smpl_volume', type=ast.literal_eval, dest='use_gt_smpl_volume', default=False, help='use_gt_smpl_volume')
        train.add_argument('--use_attention_texture', type=ast.literal_eval, dest='use_attention_texture', default=False, help='use_gt_smpl_volume')
        shuffle_train = train.add_mutually_exclusive_group()
        shuffle_train.add_argument('--shuffle_train', dest='shuffle_train', action='store_true', help='Shuffle training data')
        shuffle_train.add_argument('--no_shuffle_train', dest='shuffle_train', action='store_false', help='Don\'t shuffle training data')

        weights = self.parser.add_argument_group('Loss Weights')
        weights.add_argument('--weight_geo', type=float, default=1.0, help='weight_geo')

        optim = self.parser.add_argument_group('Optimization')
        optim.add_argument('--adam_beta1', type=float, default=0.9, help='Value for Adam Beta 1')
        optim.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
        optim.add_argument('--wd', type=float, default=0, help='Weight decay weight')

        logging = self.parser.add_argument_group('Logging')
        logging.add_argument('--debug', dest='debug', default=False, action='store_true', help='If set, debugging messages will be printed')
        logging.add_argument('--quiet', '-q', dest='quiet', default=False, action='store_true', help='If set, only warnings will be printed')
        logging.add_argument('--logfile', dest='logfile', default=None, help='If set, the log will be saved using the specified filename.')
        return

    def parse_args(self):
        """Parse input arguments."""
        self.start_time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.args = self.parser.parse_args()
        # If config file is passed, override all arguments with the values from the config file
        if self.args.from_json is not None:
            path_to_json = os.path.abspath(self.args.from_json)
            with open(path_to_json, 'r') as f:
                json_args = json.load(f)
                json_args = namedtuple('json_args', json_args.keys())(**json_args)
                return json_args
        else:
            self.args.log_dir = os.path.join(os.path.abspath(self.args.log_dir), self.args.name)
            self.args.summary_dir = os.path.join(self.args.log_dir, 'tensorboard')
            if not os.path.exists(self.args.log_dir):
                os.makedirs(self.args.log_dir)
            self.args.checkpoint_dir = os.path.join(self.args.log_dir, 'checkpoints')
            if not os.path.exists(self.args.checkpoint_dir):
                os.makedirs(self.args.checkpoint_dir)
            self.save_dump()
            return self.args

    def save_dump(self):
        """Store all argument values to a json file and create a code snapshot (useful for debugging)
        The default location is logs/expname/config_[...].json.
        """
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        with open(os.path.join(self.args.log_dir, 'config_%s.json' % self.start_time_str), 'w') as f:
            json.dump(vars(self.args), f, indent=4)
        create_code_snapshot('./', os.path.join(self.args.log_dir, 'code_bk_%s.tar.gz' % self.start_time_str))
        return
