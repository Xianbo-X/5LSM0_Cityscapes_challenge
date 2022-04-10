import torchvision.transforms.functional as TF
import random

class RandomRotation():
    """
    Rotate the given image with random angle
    """
    def __init__(self) -> None:
        pass

    def __call__(self, img_mask):
        """
        Parameters:
        ----------
            img: PIL Image or Tensor

        Returns:
        ---------
            Rotated image
        """
        # randomly generate rotation angle
        angle = random.randint(0,180)
        
        return TF.rotate(img_mask[0], angle=angle), TF.rotate(img_mask[1], angle=angle)
