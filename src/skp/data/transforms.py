import cv2
import numpy as np
import albumentations as A

from .rand_augment import RandAugment


def resize(imsize):
    x, y = imsize
    return A.Compose([
            A.LongestMaxSize(max_size=max(x,y), always_apply=True, p=1),
            A.PadIfNeeded(min_height=x, min_width=y, always_apply=True, p=1, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0)
        ], p=1)


# This can be more useful if training w/ crops
def resize_alt(imsize):
    x, y = imsize
    return A.Compose([
            A.SmallestMaxSize(max_size=max(x,y), always_apply=True, p=1)
        ], p=1)


# Ignore aspect ratio
def resize_ignore(imsize):
    x, y = imsize
    return A.Compose([
            A.Resize(imsize[0], imsize[1])
        ], p=1)


def crop(imsize, mode):
    x, y = imsize
    if mode == 'train':
        cropper = A.RandomCrop(height=x, width=y, always_apply=True, p=1)
    else:
        cropper = A.CenterCrop(height=x, width=y, always_apply=True, p=1)
    return A.Compose([
            cropper
        ], p=1, additional_targets={'image{}'.format(_) : 'image' for _ in range(1,600)})


def grayscale_augment(p, n):
    augs = A.OneOf([
        A.RandomGamma(),
        A.RandomContrast(),
        A.RandomBrightness(),
        A.ShiftScaleRotate(shift_limit=0.10, scale_limit=0, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT),
        A.ShiftScaleRotate(shift_limit=0.00, scale_limit=0.15, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT),
        A.ShiftScaleRotate(shift_limit=0.00, scale_limit=0, rotate_limit=30, border_mode=cv2.BORDER_CONSTANT),
        A.GaussianBlur(),
        A.GaussNoise()
    ], p=1)
    additional_targets = {f'image{_}' : 'image' for _ in range(1,600)}
    additional_targets.update({f'mask{_}' : 'mask' for _ in range(1,600)})
    return A.Compose([augs] * n, p=p, additional_targets=additional_targets)


def simple_augment(p=1.0):
    return A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.3, rotate_limit=15, p=0.9,
                           border_mode=cv2.BORDER_CONSTANT),
        A.OneOf([#off in most cases
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.3),
    ],
    p=p)


class Preprocessor(object):
    """
    Object to deal with preprocessing.
    Easier than defining a function.
    """
    def __init__(self, image_range, input_range, mean, sdev):
        self.image_range = image_range
        self.input_range = input_range
        self.mean = mean 
        self.sdev = sdev

    def __call__(self, img, mode='numpy'): 
        # Preprocess an input image
        image_min = float(self.image_range[0])
        image_max = float(self.image_range[1])
        model_min = float(self.input_range[0])
        model_max = float(self.input_range[1])
        image_range = image_max - image_min
        model_range = model_max - model_min 
        img = (((img - image_min) * model_range) / image_range) + model_min 

        if mode == 'numpy': 
            # Channels LAST format
            if img.shape[-1] == 3: 
            # Assume image is RGB 
            # Unconvinced that RGB<>BGR matters for transfer learning ...
                img = img[..., ::-1].astype('float32')
                img[..., 0] -= self.mean[0] 
                img[..., 1] -= self.mean[1] 
                img[..., 2] -= self.mean[2] 
                img[..., 0] /= self.sdev[0] 
                img[..., 1] /= self.sdev[1] 
                img[..., 2] /= self.sdev[2] 
            else:
                avg_mean = np.mean(self.mean)
                avg_sdev = np.mean(self.sdev)
                img -= avg_mean
                img /= avg_sdev

        elif mode == 'torch':
            # Channels FIRST format
            if img.size(1) == 3:
                img = img[:,[2,1,0]]
                img[:, 0] -= self.mean[0] 
                img[:, 1] -= self.mean[1] 
                img[:, 2] -= self.mean[2] 
                img[:, 0] /= self.sdev[0] 
                img[:, 1] /= self.sdev[1] 
                img[:, 2] /= self.sdev[2]
            else:
                avg_mean = np.mean(self.mean)
                avg_sdev = np.mean(self.sdev)
                img -= avg_mean
                img /= avg_sdev 

        return img

    def denormalize(self, img):
        # img.shape = (H, W, 3)
        img = img[..., ::-1]
        img[..., 0] *= self.sdev[0]
        img[..., 1] *= self.sdev[1]
        img[..., 2] *= self.sdev[2]
        img[..., 0] += self.mean[0]
        img[..., 1] += self.mean[1]
        img[..., 2] += self.mean[2]

        image_min = float(self.image_range[0])
        image_max = float(self.image_range[1])
        model_min = float(self.input_range[0])
        model_max = float(self.input_range[1])
        image_range = image_max - image_min
        model_range = model_max - model_min 

        img = ((img - model_min) * image_range) / model_range + image_min 
        return img

