import argparse
import numpy as np
import matplotlib.pyplot as plt
import celestine.utils as utils
from celestine.imitator import *

parser = argparse.ArgumentParser(description="ZXX TRAIN IMITATOR")
parser.add_argument(
    "--renderer",
    type=str,
    default="oilpaintbrush",
    metavar="str",
    help="renderer: [watercolor, markerpen, oilpaintbrush, rectangle, bezier, circle, square, rectangle]",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    metavar="N",
    help="input bash size of training (default 4)",
)
parser.add_argument(
    "--print_models",
    action="store_true",
    default=False,
    help="visualize and print networks",
)
parser.add_argument(
    "--checkopoint_dir",
    type=str,
    default=r"./checkpoints_G",
    metavar="str",
    help="directory to save checkpoints",
)
parser.add_argument(
    "--vis_dir",
    type=str,
    default=r"./val_out_G",
    metavar="str",
    help="directory to save results during training (default ``val_out_G``)",
)
parser.add_argument(
    "--lr", type=float, default=2e-4, help="learning rate (default 0.0002)"
)
parser.add_argument(
    "--max_num_epochs",
    type=int,
    default=400,
    metavar="N",
    help="max number of training epochs",
)
args = parser.parse_args()

if __name__ == "__main__":
    dataloaders = utils.get_renderer_loaders(args)
    imt = Imitator(args=args, dataloaders=dataloaders)
    imt.train_models()
