import random
import torchvision.transforms.functional as TF
import torch

class RandomCropAndResize():
    """
    RandomCropThe Image and resize
    """
    def __init__(self,pr=0.5,width_lim=None,height_lim=None) -> None:
        self.pr=pr
        self.lower_width=5 if width_lim is None else width_lim[0]
        self.lower_height=5 if height_lim is None else height_lim[0]
        self.upper_width=20 if width_lim is None else width_lim[1]
        self.upper_height=20 if height_lim is None else height_lim[1]

    def __call__(self, img_mask) -> None:
        res=img_mask
        if random.random()<self.pr:
            shape=(res[0].size[1],res[0].size[0]) # height \times width
            width=random.randint(self.lower_width,min(shape[1],self.upper_width))
            height=random.randint(self.lower_height,min(shape[0],self.upper_height))
            corner_left=random.randint(0,shape[1]-width)
            corner_top=random.randint(0,shape[0]-height)
            
            # corner_left=0
            # corner_top=0
            print((corner_left,corner_top))
            print((width,height))
            f=lambda x: TF.resized_crop(x,corner_top,corner_left,height,width,size=shape) if x is not None else None
            res=(f(res[0]),f(res[1]))
        return res