import os
import argparse
import torch
from pathlib import Path
import pprint

def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)

    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--embedding_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=400)
    parser.add_argument('--bidirectional', type=str2bool, default=False)

    # Utility
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--plot_every_epoch', type=int, default=1)
    parser.add_argument('--save_every_epoch', type=int, default=1)

    parser.add_argument('--path', type=str, default='./model/')

    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)
