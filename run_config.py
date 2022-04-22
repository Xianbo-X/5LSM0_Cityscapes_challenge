import os
from tqdm import tqdm
import numpy as np

if __name__=="__main__":
    run_name=["unet_1","unet_2","unet_2_1","res_unet","pos_unet"]

    cmds=[]
    for  model in run_name:
        folder=os.path.join("config",model)
        configurations=map(lambda x:os.path.join(folder,x),os.listdir(folder))
        cmds_one_model=[]
        for config in configurations:
            if "template" in config: continue
            cmd=f"python main.py --path {config}"
            cmds_one_model.append(cmd)
            # print(cmd)
            # os.system(cmd)
        cmds.append(cmds_one_model)
    cmds=np.array(cmds).T.flatten()
    for cmd in tqdm(cmds):
        os.system(cmd)
