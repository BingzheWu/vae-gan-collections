
import torch
import numpy as np
from PIL import Image
import inspect

def tensor2im(image_tensor, imtype = np.uint8, cv2_rgb = True):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1 and cv2_rgb:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1,2,0))+1) /2.0*255.0
    return image_numpy.astype(imtype)
