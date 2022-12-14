import numpy as np
import skimage
import cv2

def psnr(pred: np.ndarray, target: np.ndarray):
    target = cv2.resize(target, (pred.shape[1], pred.shape[0]))
    mse: np.float64 = np.mean(np.square(target - pred))
    max_f = 255
    return 10 * np.log10(max_f / mse)