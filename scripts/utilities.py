from matplotlib import pyplot as plt
import numpy as np
import os
import torch
import torchvision.transforms.functional as TF

IMAGE_DIR = os.path.abspath("fmap_visualization")

def plot(imgs, **imshow_kwargs):
    """
    ref: https://pytorch.org/vision/stable/auto_examples/plot_optical_flow.html#sphx-glr-auto-examples-plot-optical-flow-py
    """
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()



def visualize_fmap(image, model):
    """
    Visualize the intermediate fmap for analysis
    """

    # 1. Visualization of fmap after inc layer
    with torch.no_grad():
        out_fmap = model.inc(image.unsqueeze(0)).squeeze().detach()
    
    for i in range(64):
        img=TF.to_pil_image(out_fmap[i])
        img.save(os.path.join(IMAGE_DIR,"inc_fmap_"+str(i)+".jpg"))

    # 2. Visualization of fmap after down1 layer
    with torch.no_grad():
        down1_fmap = model.down1(out_fmap.unsqueeze(0)).squeeze().detach()

    for i in range(10):
        img=TF.to_pil_image(down1_fmap[i])
        img.save(os.path.join(IMAGE_DIR,"down1_fmap_"+str(i)+".jpg"))


    # 3. Visualization of fmap after down4 layer(bottleneck)
    with torch.no_grad():
        down2_fmap = model.down2(down1_fmap.unsqueeze(0)).squeeze().detach()
        down3_fmap = model.down3(down2_fmap.unsqueeze(0)).squeeze().detach()
        down4_fmap = model.down4(down3_fmap.unsqueeze(0)).squeeze().detach()

    for i in range(10):
        img=TF.to_pil_image(down4_fmap[i])
        img.save(os.path.join(IMAGE_DIR,"down4_fmap_"+str(i)+".jpg"))

