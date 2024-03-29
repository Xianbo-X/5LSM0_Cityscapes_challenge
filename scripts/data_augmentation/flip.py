import torchvision.transforms.functional as TF
import random

class RandomHorizontalFlip():
    """
    Flip the given image horizontally,
    """

    def __init__(self,pr=0.5) -> None:
        pass

    def __call__(self, img):
        """
        Parameters:
        ----------
            img: PIL Image or Tensor

        Returns:
        ---------
            Flipped image
        """
        if random.random()<self.pr:
            return TF.hflip(img)
        return img

class RandomVerticalFlip():
    """
    Flip the given image vertically
    """

    def __init__(self,pr=0.5) -> None:
        pass

    def __call__(self, img):
        """
        Parameters:
        ----------
            img: PIL Image or Tensor

        Returns:
        ---------
            Flipped image
        """
        if random.random()< self.pr:
            return TF.vflip(img)
        return img
        
class RandomHorizontalFlip2():
    """
    Flip the given image and mask horizontally,
    """

    def __init__(self,pr=0.5) -> None:
        self.pr=pr

    def __call__(self, img_mask):
        """
        Parameters:
        ----------
            img: PIL Image or Tensor
            mask: PIL Image or Tensor

        Returns:
        ---------
            Flipped image, Flipped mask
        """
        res=img_mask
        if random.random()< self.pr:
            res=(TF.hflip(img_mask[0]),TF.hflip(img_mask[1]))
        return res

class RandomVerticalFlip2():
    """
    Flip the given image vertically
    """

    def __init__(self,pr=0.5) -> None:
        self.pr=pr

    def __call__(self, img_mask):
        """
        Parameters:
        ----------
            img: PIL Image or Tensor
            mask: PIL Image or Tensor

        Returns:
        ---------
            Flipped image, Flipped mask
        """
        res=img_mask
        if random.random()< self.pr:
            res=(TF.vflip(img_mask[0]),TF.vflip(img_mask[1]))
        return res