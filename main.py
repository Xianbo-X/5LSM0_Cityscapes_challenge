# %%
import torch
import os
import argparse
import sys
import argparse

from scripts.utilities.parse_config import Config
from scripts.dataset.dataset import CityscapesDataset_Aug,classes
from scripts.metrics import IoU,compute_iou
from scripts.Trainer import Trainer

dir_data = os.path.abspath("data")

# URLs to retrieve ground truth and images data from. 
url_truth = 'https://flux127120.nbw.tue.nl/index.php/s/Cwxa5Ft2pQBK9N7/download'
dir_truth = os.path.join(dir_data, "gtFine")

url_input = 'https://flux127120.nbw.tue.nl/index.php/s/Tz3GCjQwwsiHgqC/download'
dir_input = os.path.join(dir_data, "leftImg8bit")

# Target size of each sample in the dataset
sample_size = (256, 128)

# Directories for preprocessed datasets
dir_truth_pp, dir_input_pp = (f'{d}_{sample_size[0]}_{sample_size[1]}' for d in (dir_truth, dir_input))



if __name__=="__main__":
    parser=argparse.ArgumentParser("Train model")
    parser.add_argument(
        '--path',
        help='provide config file path',
        default=None
    )
    parser.add_argument(
        '--save_name',
        help='name to save the model',
        default=None
    )
    args=parser.parse_args()
    if args.path is not None:
        config_path=os.path.abspath(args.path)
        print(f"load from {config_path}")
        conf=Config(config_path)
        # Create one instance of the CityscapesDataset for each split type
        ds_split = {
            name:CityscapesDataset_Aug(os.path.join(dir_input_pp, name), os.path.join(dir_truth_pp, name), sample_size, classes)
            for name in ("train", "val", "test")
        }
        for key,value in ds_split.items():
            value.set_transform_list(conf.get_transformationlist())

        ds_split["val"].no_aug()
        ds_split["test"].no_aug()

        model = conf.get_model()
        trainer=Trainer(model,ds_split,**conf.get_func_param(Trainer.__init__),writer=None)
        trainer.fit(**conf.get_func_param(trainer.fit))
        if args.save_name is not None:
            torch.save(model,args.save_name)