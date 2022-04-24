from ast import parse
import os
from tqdm import tqdm
import numpy as np
import argparse

if __name__=="__main__":
    # run_name=["unet_1","unet_2","unet_2_1","res_unet","pos_unet"]
    parser=argparse.ArgumentParser("Run multiple configs")
    parser.add_argument("--config",help="config folder name",nargs="+")
    args=parser.parse_args()
    print(args.config)
    cmds=[]
    for  folder in args.config:
        # folder=os.path.join("config",model)
        configurations=map(lambda x:os.path.join(folder,x),os.listdir(folder))
        cmds_one_model=[]
        for config in configurations:
            if "template" in config: continue
            cmd=f"python main.py --path {config}"
            cmds_one_model.append(cmd)
            # print(cmd)
    #         # os.system(cmd)
        cmds.append(cmds_one_model)
    # print(cmds)
    cmds=np.array(cmds).T.flatten()
    for cmd in tqdm(cmds):
        os.system(cmd)