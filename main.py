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
import json

from torch.utils.tensorboard import SummaryWriter

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

def training(model,ds_split,conf:Config,writer):
    # Train the network
    print("Testing training process...")
    writer = SummaryWriter(os.path.join(result_folder,"logs/passthrough"))
    trainer = Trainer(model, ds_split,writer=writer, **conf.get_func_param(Trainer.__init__))
    df_train, df_val = trainer.fit(result_folder=conf.get_result_folder(),model_path_prefix=conf.get_saves_path_prefix(),**conf.get_func_param(trainer.fit))

    with open(os.path.join(result_folder,"train.json"), 'w') as file_train:
        json.dump(df_train.to_dict(),file_train)
    with open(os.path.join(result_folder,"val.json"), 'w') as file_val:
        json.dump(df_val.to_dict(),file_val)

    

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
        
        path = conf.conf["path"]
        SAVE_DIR = os.path.abspath(path["ROOT_PATH"])
        
        model_folder=conf.get_model_folder()
        result_folder=conf.get_result_folder()

        assert not os.path.exists(model_folder)
        os.makedirs(model_folder)
        assert not os.path.exists(result_folder)
        os.makedirs(result_folder)
        print(f"model foler: {model_folder}")
        print(f"result foler: {result_folder}")
        try:
            model = conf.get_model()
            print("model name: "+str(model).split("\n")[0])
            training(model,ds_split,**conf.get_func_param(training),result_folder=result_folder)
            # trainer=Trainer(model,ds_split,**conf.get_func_param(Trainer.__init__),writer=None)
            # trainer.fit(**conf.get_func_param(trainer.fit))
            if args.save_name is not None:
                torch.save(model,args.save_name)
        except Exception as e:
            torch.save(model,os.path.join(model_folder,"failed_autosave_model.pt"))
            raise(e)