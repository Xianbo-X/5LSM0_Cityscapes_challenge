import torchvision.transforms.functional as TF

class Invert():
    """
    Invert the colors of the given image
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
            Inverted image
        """
        
        return TF.invert(img_mask[0]), img_mask[1]
