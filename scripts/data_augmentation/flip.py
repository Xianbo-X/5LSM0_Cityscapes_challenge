import torchvision.transforms.functional as TF

class HorizontalFlip():
    """
    Flip the given image horizontally,
    """

    def __init__(self) -> None:
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
        return TF.hflip(img)

class VerticalFlip():
    """
    Flip the given image vertically
    """

    def __init__(self) -> None:
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
        return TF.vflip(img)
        