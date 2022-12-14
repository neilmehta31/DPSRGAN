import argparse
import os
import sys
import cv2
from utils.metrics import psnr as PSNR
from pytorch_lightning import Trainer
from RealESRGAN import RealESRGAN
from PIL import Image
import numpy as np
import torch
from src.models import SRGAN, SRResNet
from utils.dataloader import SRDataLoader

parser = argparse.ArgumentParser(prog="Testing script",
                                 description="A script for testing out trained models on a set of test images")
parser.add_argument("--model_path", type=str,
                    help='path to the pretrained model checkpoint file')
parser.add_argument("--data_dir", type=str,
                    help='path to directory where images are stored')
parser.add_argument("--network", type=str, choices=["SRGAN", "SRResNet"], default="SRGAN",
                    help="type of network, either GAN or SRResNet")
parser.add_argument('--use_single_img', type=str, default=False)

args = parser.parse_args()


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            print(f"Loaded image: {filename}")
    return images

def calc_snr_ssim(pred,hr_img):
    psnr = PSNR(pred, hr_img)
    print(f"PSNR: {psnr}")


def process_input(lr_img,hr_img):
    result_image_path = os.path.join('pred2/',os.path.basename(lr_img))
    image = Image.open('data/lr_test/' + lr_img).convert('RGB')
    sr_image = model.predict(np.array(image))
    calc_snr_ssim(sr_image,hr_img)
    sr_image.save(result_image_path)    
    print(f'Finished! Image saved to {result_image_path}')


if __name__ == "__main__":
    if args.use_single_img==False:
        print("Testing on a set of images")
        if not args.model_path:
            print("Model path needs to be specified!")
            sys.exit(1)
        try:
            os.mkdir("preds")
        except FileExistsError:
            pass
        if args.network == "SRGAN":
            model = SRGAN.load_from_checkpoint(args.model_path)
        else:
            model = SRResNet.load_from_checkpoint(args.model_path)
        data = SRDataLoader(data_dir=args.data_dir, batch_size=1)
        data.setup('test')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('device:', device)
        model = RealESRGAN(device, scale=int(2))
        model.load_weights('models/epoch=18_2x.pth')


        # For testing on a set of images 
        hr_test = load_images_from_folder('data/test')
        lr_test = 'data/lr_test'
        # print(data.test)
        for hr_index,lr_img in enumerate(os.listdir(lr_test)):
            if hr_index>20000:
                break
            hr_img = hr_test[hr_index]
            print('Processing:> low res Img :', lr_img)
            process_input(lr_img,hr_img)
    else :
        print("Testing on a single image")
    
        # For testing on a single image
        lr_img_name = 'interpolated_2.jpg'
        hr_img_name = '180002.jpg'
        test_lr_img = Image.open('data/lr_test/' + lr_img_name).convert('RGB')
        test_hr_img = Image.open('data/test/' + hr_img_name).convert('RGB')
        calc_snr_ssim(np.array(test_lr_img),np.array(test_hr_img))

    

   

