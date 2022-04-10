import torchvision.transforms.functional as TF
import random

class Rgb2Gray():
    """
    Convert the RGB image to grayscale Image with given channel number.
    """

    def __init__(self,channels=3) -> None:
        """
        Parameters:
        ----------
            channels: int
                Output image channels

        Returns:
        ---------
            Instance of Rgb2Gray class
        """
        self.channels=int(channels)
    
    def __call__(self, img):
        """
        Parameters:
        ----------
            img: PIL Image or Tensor

        Returns:
        ---------
            Grayscaled image
        """
        return TF.rgb_to_grayscale(img,self.channels)

class RandomRgb2Gray2():
    """
    Convert the RGB image to grayscale Image with given channel number.
    """

    def __init__(self,pr=0.5,channels=3) -> None:
        """
        Parameters:
        ----------
            channels: int
                Output image channels

        Returns:
        ---------
            Instance of Rgb2Gray class
        """
        self.channels=int(channels)
        self.pr=pr
    
    def __call__(self, img_mask):
        """
        Parameters:
        ----------
            img: PIL Image or Tensor

        Returns:
        ---------
            Grayscaled image
        """
        if random.random()<self.pr:
            return (TF.rgb_to_grayscale(img_mask[0],self.channels),img_mask[1])
        else:
            return img_mask