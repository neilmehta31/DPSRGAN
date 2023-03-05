import os
import shutil
from pathlib import Path

from PIL import Image
import torch
from torchvision.transforms import Resize, ToTensor
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
import requests
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
import torchvision

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


class SRDataset(Dataset):
    def __init__(self, data_dir: str = 'images/train/', img_size: int = 128):
        super(SRDataset, self).__init__()
        self.img_dir = data_dir
        self.filenames = os.listdir(self.img_dir)
        self.toTensor = ToTensor()
        self.setSize = Resize((img_size, img_size),
                              interpolation=Image.BICUBIC)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index, save = False):
        img = Image.open(os.path.join(self.img_dir, self.filenames[index]))
        # print(np.shape(img))
        img = self.setSize(img)
        lr_img = recursiveResize(img, 2)
        hr_img = img
        interpolated_img = self.setSize(lr_img)
        if save:
            interpolated_img.save(f'C:/Work/Super-Resolution/data/paper_pics/lr_test/interpolated_{index}.jpg')
        lr_img, hr_img, interpolated_img = self.toTensor(
            lr_img), self.toTensor(hr_img), self.toTensor(interpolated_img)
        # interpolated_img = interpolated_img.reshape(128,128,3)
        return lr_img, hr_img, interpolated_img


class SRDataLoader(LightningDataModule):
    def __init__(self, url: str = "", data_dir: str = "images/", batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.url = url

        self.data_dir = Path(data_dir)
        self.train_dir = Path(os.getcwd(), self.data_dir, 'train')
        self.test_dir = Path(os.getcwd(), self.data_dir, 'lr_test')
        self.val_dir = Path(os.getcwd(), self.data_dir, 'val')
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)

        self.img_size = 128
        self.train, self.val, self.test = None, None, None

    def prepare_data(self):
        if self.url != "":
            self.download_data(self.url)
            self.split_data()

    def download_data(self, url: str):
        req = requests.get(url)
        with open(f'{self.data_dir}', 'wb+') as data_dir:
            data_dir.write(req.content)

    def split_data(self):
        images = os.listdir(self.data_dir)
        train = images[:int(0.8 * len(images))]
        test = images[int(0.8 * len(images)):]
        os.chdir(self.data_dir)
        for img in train:
            shutil.copy(img, self.train_dir)
        for img in test:
            shutil.copy(img, self.test_dir)

    def setup(self, stage=None):
        if stage == "fit":
            # print("\n\n\n",len(SRDataset(data_dir=self.train_dir, img_size=self.img_size)),"\n\n\n","self.train_dir = ",self.train_dir,"\n\n\n")
            # self.train, self.val = random_split(
            #     SRDataset(data_dir=self.train_dir, img_size=self.img_size), lengths=[180000, 2059],
            #     generator=torch.Generator().manual_seed(0))
            self.train, self.val = SRDataset(data_dir=self.train_dir,img_size=self.img_size), SRDataset(data_dir=self.val_dir,img_size=self.img_size)
            # lr_tensor_img = self.train.__getitem__(0)[0]
            # hr_tensor_img = self.train.__getitem__(0)[1]
            # interpolated_tensor_img = self.train.__getitem__(0)[2]

            # plt.imshow(lr_tensor_img[0])
            # plt.imshow(hr_tensor_img[0])
            # plt.imshow(interpolated_tensor_img[0])
            print('len of train : ',len(self.train),' len of val : ',len(self.val))
        elif stage == 'test':
            self.test = SRDataset(data_dir=self.test_dir,
                                  img_size=self.img_size)
            print(len(self.test))
            for index in range(len(self.test)):
                save = self.test.__getitem__(index,save = True)
                # interpolated_tensor_img = self.test.__getitem__(index)[2][0]
                # save_image(interpolated_tensor_img, f'C:/Work/Super-Resolution/data/paper_pics/interpolated_{index}.jpg')
            # for index,test_img in enumerate(self.test):
            #     print(index, test_img.__getitem__(index)[2])
            #     interpolated_tensor_img = test_img.__getitem__(index)[2]
            #     plt.imshow(interpolated_tensor_img[0])
            #     # interpolated_tensor_img.save(f'C:/Work/Super-Resolution/data/lr_test/interpolated_{index}.jpg')
            #     save_image(interpolated_tensor_img, f'C:/Work/Super-Resolution/data/lr_test/interpolated_{index}.jpg')
            print("done")

    def train_dataloader(self, *args, **kwargs):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=4, drop_last=True,
                          pin_memory=True)

    def val_dataloader(self, *args, **kwargs):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=4, pin_memory=True, drop_last=True)

    def test_dataloader(self, *args, **kwargs):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=4, pin_memory=True, drop_last=True)
