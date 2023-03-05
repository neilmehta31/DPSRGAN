from PIL import Image
from torchvision.transforms import Resize, ToTensor
import numpy as np
def recursiveResize(img: Image, factor: int = 2):
    """
    Recursively resizes an image by down scaling by 2,
    repeats this for factor times
    Args:
        img (PIL.Image): image to be resized
        factor (int): factor by which resizing is to take place. eg. if factor is 2, image will be downscaled
        to half it's size twice, thereby final image with be 1/4th the original image size
    Returns:

    """
    for _ in range(factor):
        height, width = img.size
        print(height, width)
        resize = Resize((int(height / 2), int(width / 2)),
                        interpolation=Image.BICUBIC)
        img = resize(img)
    return img

def interppolate(filename, save=True):
    print(filename)
    img = Image.open(filename)
    print(np.shape(img))
    img = Resize((128,128),interpolation=Image.BICUBIC)
    lr_img = recursiveResize(img, 2)
    hr_img = img
    interpolated_img = Resize((128,128),interpolation=Image.BICUBIC)
    if save:
        print('saving')
        interpolated_img.save(f'C:/Work/Super-Resolution/data/interpolated_{index}.jpg')

if __name__ == "__main__":
    interppolate('C:/Work/Super-Resolution/data/000004.jpg')
    interppolate('C:/Work/Super-Resolution/data/000007.jpg')
    interppolate('C:/Work/Super-Resolution/data/000009.jpg')