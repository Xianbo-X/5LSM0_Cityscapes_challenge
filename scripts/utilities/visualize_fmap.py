"""
Visualize the intermediate fmap for analysis
"""

import os
import torch
import torchvision.transforms.functional as TF

IMAGE_DIR = os.path.abspath("fmap_visualization")

def visualize_fmap(image, model):

    # 1. Visualization of fmap after inc layer
    with torch.no_grad():
        out_fmap = model.inc(image.unsqueeze(0)).squeeze().detach()
    
    for i in range(64):
        img=TF.to_pil_image(out_fmap[i])
        img.save(os.path.join(IMAGE_DIR,"inc_fmap"+str(i)+".png", 'PNG'))

    # 2. Visualization of fmap after down1 layer
    with torch.no_grad():
        down1_fmap = model.down1(out_fmap.unsqueeze(0)).squeeze().detach()

    for i in range(128):
        img=TF.to_pil_image(down1_fmap[i])
        img.save(os.path.join(IMAGE_DIR,"down1_fmap"+str(i)+".png", 'PNG'))

