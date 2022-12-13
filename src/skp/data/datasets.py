import albumentations as A
import cv2
import glob
import random
import numpy as np
import os, os.path as osp
import pydicom
import torch
import torch.nn.functional as F

from scipy.ndimage.interpolation import zoom
from torch.utils import data


NONETYPE = type(None)


def augmentations(p=0.9, N=3):
    auglist = A.OneOf([
        A.ShiftScaleRotate(shift_limit=0.10, scale_limit=0, rotate_limit=0, border_mode=cv2.BORDER_REPLICATE),
        A.ShiftScaleRotate(shift_limit=0.00, scale_limit=0.15, rotate_limit=0, border_mode=cv2.BORDER_REPLICATE),
        A.ShiftScaleRotate(shift_limit=0.00, scale_limit=0, rotate_limit=30, border_mode=cv2.BORDER_REPLICATE),
    ], p=1)
    return A.Compose([auglist] * N, p=p)


class ImageDataset(data.Dataset):
    def __init__(
        self,
        inputs,
        labels,
        resize=None,
        augment=None,
        crop=None,
        preprocess=None,
        flip=False,
        verbose=True,
        test_mode=False,
        return_name=False,
        return_imsize=False,
        invert=False,
        add_invert_label=False,
        repeat_rgb=True,
        **kwargs
    ):
        self.inputs = inputs
        self.labels = labels
        self.resize = resize
        self.augment = augment
        self.crop = crop
        self.preprocess = preprocess
        self.flip = flip
        self.verbose = verbose
        self.test_mode = test_mode
        self.return_name = return_name
        self.return_imsize = return_imsize
        self.invert = invert
        self.add_invert_label = add_invert_label
        self.repeat_rgb = repeat_rgb

    def __len__(self):
        return len(self.inputs)

    def process_image(self, X):
        if self.resize:
            X = self.resize(image=X)["image"]
        if self.augment:
            X = self.augment(image=X)["image"]
        if self.crop:
            X = self.crop(image=X)["image"]
        if self.invert:
            X = np.invert(X)
        if self.preprocess:
            X = self.preprocess(X)
        return X.transpose(2, 0, 1)

    @staticmethod
    def flip_array(X):
        # X.shape = (C, H, W)
        if random.random() > 0.5:
            X = X[:, :, ::-1]
        if random.random() > 0.5:
            X = X[:, ::-1, :]
        if random.random() > 0.5 and X.shape[-1] == X.shape[-2]:
            X = X.transpose(0, 2, 1)
        X = np.ascontiguousarray(X)
        return X

    def get(self, i):
        try:
            if self.repeat_rgb:
                X = cv2.imread(self.inputs[i])
            else:
                X = cv2.imread(self.inputs[i], 0)
                if not isinstance(X, NONETYPE):
                    X = np.expand_dims(X, axis=-1)
            return X
        except Exception as e:
            if self.verbose:
                print(e)
            return None

    def __getitem__(self, i):
        X = self.get(i)
        while isinstance(X, NONETYPE):
            if self.verbose:
                print("Failed to read {} !".format(self.inputs[i]))
            i = np.random.randint(len(self))
            X = self.get(i)

        imsize = X.shape[:2]

        if self.add_invert_label:
            inverted = False
            if np.random.binomial(1, 0.5):
                X = 255 - X
                inverted = True

        X = self.process_image(X)

        if self.flip and not self.test_mode:
            X = self.flip_array(X)

        y = self.labels[i]

        if self.add_invert_label:
            if inverted:
                y = np.concatenate([y, np.asarray([1])])
            else:
                y = np.concatenate([y, np.asarray([0])])

        X = torch.tensor(X).float()
        y = torch.tensor(y)

        out = [X, y]
        if self.return_name:
            out.append(self.inputs[i])
        if self.return_imsize:
            out.append(imsize)

        return tuple(out)


class ImageStackDataset(data.Dataset):

    def __init__(self,
                 inputs,
                 labels,
                 num_slices=32,
                 repeat_3ch=True,
                 resize=None,
                 augment=None,
                 crop=None,
                 preprocess=None,
                 flip=False,
                 verbose=True,
                 test_mode=False):
        self.inputs = inputs
        self.labels = labels
        self.num_slices = num_slices 
        self.repeat_3ch = repeat_3ch
        self.resize = resize 
        self.augment = augment 
        self.crop = crop 
        self.preprocess = preprocess 
        self.flip = flip 
        self.verbose = verbose 
        self.test_mode = test_mode 

    def __len__(self):
        return len(self.inputs)

    @staticmethod
    def flip_array(X):
        # X.shape = (C, Z, H, W)
        if random.random() > 0.5:
            X = X[:, :, :, ::-1]
        if random.random() > 0.5:
            X = X[:, :, ::-1, :]
        if random.random() > 0.5:
            X = X[:, ::-1, :, :]
        X = np.ascontiguousarray(X)
        return X

    def resample_slices(self, X):
        new_size = (self.num_slices, X.shape[-2], X.shape[-1])
        X = F.interpolate(X.unsqueeze(0), size=new_size, mode='trilinear', align_corners=False)
        return X.squeeze(0)

    def process_image(self, X):
        if self.resize: 
            X = np.asarray([self.resize(image=_)['image'] for _ in X])
        if X.ndim == 3: X = np.expand_dims(X, axis=-1)
        if self.repeat_3ch: X = np.repeat(X, 3, axis=-1)
        assert X.ndim == 4
        # X.shape (Z, H, W, C)
        if self.augment and not self.test_mode: 
            to_augment = {'image{}'.format(ind) : X[ind] for ind in range(1,len(X))}
            to_augment.update({'image': X[0]})
            augmented = self.augment(**to_augment)
            X = np.asarray([augmented['image']] + [augmented['image{}'.format(_)] for _ in range(1,len(X))])
        if self.crop: 
            to_crop = {'image{}'.format(ind) : X[ind] for ind in range(1,len(X))}
            to_crop.update({'image': X[0]})
            cropped = self.crop(**to_crop)
            X = np.asarray([cropped['image']] + [cropped['image{}'.format(_)] for _ in range(1,len(X))])
        if self.preprocess: X = self.preprocess(X)
        return X.transpose(3, 0, 1, 2)

    @staticmethod
    def check_if_image(fp):
        return "png" in fp or "jpg" in fp or "jpeg" in fp

    def get(self, i):
        try:
            slices = np.sort(glob.glob(osp.join(self.inputs[i], '*')))
            slices = [s for s in slices if self.check_if_image(s)]
            if len(slices) == 0:
                return None
            indices_to_load = zoom(np.arange(len(slices)), float(self.num_slices) / len(slices), order=0, prefilter=False)
            slices = np.asarray([cv2.imread(slices[ind_load]) for ind_load in indices_to_load])
            assert len(slices) == self.num_slices
            return slices
        except Exception as e:
            if self.verbose:
                print(f'Failed to load {self.inputs[i]} :  {e}')
            return None 

    def __getitem__(self, i):
        X = self.get(i)
        while isinstance(X, NONETYPE):
            print(f"Failed to read {self.inputs[i]} !")
            i = np.random.randint(len(self))
            X = self.get(i) 

        X = self.process_image(X) 

        if self.flip and not self.test_mode:
            X = self.flip_array(X) 

        X = torch.tensor(X).float()
        if X.shape[1] != self.num_slices:
            X = self.resample_slices(X) 

        y = torch.tensor(self.labels[i])
        return X, y 


class DICOMDataset(ImageDataset):

    def __init__(self, *args, **kwargs):
        self.window = kwargs.pop("window", [100,700])
        super().__init__(*args, **kwargs)


    @staticmethod
    def load_dicom(dcmfile):
        try: 
            dicom = pydicom.dcmread(dcmfile)
            for at in ["pixel_array", "RescaleSlope", "RescaleIntercept", "ImagePositionPatient"]:
                assert hasattr(dicom, at)
            return dicom
        except Exception as e:
            print(f"DICOM READ ERROR : Failed to load {dcmfile} // {e}")
            return None 


    def apply_window(self, array):
        WL, WW = self.window
        WL, WW = float(WL), float(WW)
        lower, upper = WL - WW / 2, WL + WW / 2
        array = np.clip(array, lower, upper) 
        array = array - lower
        array = array / upper 
        array = array * 255.0
        return array.astype("uint8")


    def get(self, i):
        try:
            dicom = self.load_dicom(self.inputs[i])
            rescale_intercept = float(dicom.RescaleIntercept)
            rescale_slope = float(dicom.RescaleSlope)
            dicom = dicom.pixel_array.astype("float32")
            dicom = dicom * rescale_slope + rescale_intercept
            dicom = self.apply_window(dicom)
            dicom = np.expand_dims(dicom, axis=-1)
            if self.repeat_rgb:
                dicom = np.repeat(dicom, 3, axis=-1)
            return dicom
        except Exception as e:
            if self.verbose:
                print(f'Failed to load {self.inputs[i]} :  {e}')
            return None 


class DICOMStackDataset(ImageStackDataset):

    def __init__(self, *args, **kwargs):
        self.window = kwargs.pop("window", [100,700])
        super().__init__(*args, **kwargs)


    @staticmethod
    def load_dicom(dcmfile):
        try: 
            dicom = pydicom.dcmread(dcmfile)
            for at in ["pixel_array", "RescaleSlope", "RescaleIntercept", "ImagePositionPatient"]:
                assert hasattr(dicom, at)
            return dicom
        except Exception as e:
            print(f"DICOM READ ERROR : Failed to load {dcmfile} // {e}")
            return None 


    def apply_window(self, array):
        WL, WW = self.window
        WL, WW = float(WL), float(WW)
        lower, upper = WL - WW / 2, WL + WW / 2
        array = np.clip(array, lower, upper) 
        array = array - lower
        array = array / upper 
        array = array * 255.0
        return array.astype("uint8")


    def get(self, i):
        try:
            # Assumes all files in the folder are readable image DICOM files
            dicoms = glob.glob(osp.join(self.inputs[i], '*'))
            dicoms = [self.load_dicom(dcm) for dcm in dicoms]
            dicoms = [dcm for dcm in dicoms if not isinstance(dcm, type(None))]
            if len(dicoms) == 0:
                print(f"Failed to read {self.inputs[i]} : no valid DICOM files")
                return None
            # Assumes images in axial plane and orders by z-position
            z_positions = [float(dcm.ImagePositionPatient[2]) for dcm in dicoms]
            z_positions = np.argsort(z_positions)
            rescale_intercept = float(dicoms[0].RescaleIntercept)
            rescale_slope = float(dicoms[0].RescaleSlope)
            dicoms = np.asarray([dcm.pixel_array.astype('float32') for dcm in dicoms])
            dicoms = dicoms[z_positions]
            dicoms = dicoms * rescale_slope + rescale_intercept
            dicoms = self.apply_window(dicoms)
            return dicoms
        except Exception as e:
            if self.verbose:
                print(f'Failed to load {self.inputs[i]} :  {e}')
            return None 


class NumpySliceDataset(data.Dataset):

    def __init__(self,
                 inputs,
                 labels,
                 input_size=(224, 224),
                 add_noise=0.01,
                 flip=True,
                 verbose=True,
                 test_mode=False):
        self.inputs = inputs
        self.labels = labels
        self.input_size = input_size
        self.add_noise = add_noise
        self.flip = flip
        self.verbose = verbose
        self.test_mode = test_mode

    def __len__(self): return len(self.inputs)

    @staticmethod
    def flip_array(X):
        # X.shape = (C, H, W)
        if random.random() > 0.5:
            X = X[:, :, ::-1]
        if random.random() > 0.5:
            X = X[:, ::-1, :]
        if random.random() > 0.5:
            X = X[::-1, :, :]
        X = np.ascontiguousarray(X)
        return X

    @staticmethod
    def get_pad_dims(actual, desired):
        diff = desired - actual
        pad1 = diff // 2
        pad2 = diff - pad1
        return (pad1, pad2)

    def pad_array(self, array, size):
        # array.shape = (C, H, W)
        h_pad = self.get_pad_dims(array.shape[1], size[0])
        w_pad = self.get_pad_dims(array.shape[2], size[1])
        array = np.pad(array,[(0,0), h_pad, w_pad], mode='constant', 
                       constant_values=np.min(array))
        return array

    def pad_and_resize(self, X):
        X = self.pad_array(X, self.input_size)
        return X

    def resample_slices(self, X):
        X = F.interpolate(X.unsqueeze(0), size=tuple(self.input_size), mode='bilinear')
        return X.squeeze(0)

    def get(self, i):   
        # try:
            X = np.stack([np.load(each_file) for each_file in self.inputs[i]])
            return X
        # except Exception as e:
        #     if self.verbose: print(e)
        #     return None

    def __getitem__(self, i):
        X = self.get(i)
        while isinstance(X, NONETYPE):
            i = np.random.uniform(len(self))
            X = self.get(i)

        X = self.pad_and_resize(X)

        # if not self.test_mode:
        #     c, z, h, w = X.shape
        #     X = X.reshape(c*z, h, w)
        #     to_augment = {'image': X[0]}
        #     for i in range(1, len(X)):
        #         to_augment[f'image{i}'] = X[i]
        #     augmented = self.augment(**to_augment)
        #     X = np.concatenate([augmented['image'] if i == 0 else augmented[f'image{i}'] for i in range(len(X))])
        #     X = X.reshape(c, z, h, w)

        if self.add_noise > 0 and not self.test_mode:
            noise = np.random.normal(0, self.add_noise, X.shape)
            X[X > np.min(X)] += noise[X > np.min(X)]

        if self.flip and not self.test_mode:
            X = self.flip_array(X)

        X = np.ascontiguousarray(X)
        X = torch.from_numpy(X).float()
        y = torch.tensor(self.labels[i])

        if X.shape[1] > self.input_size[0]:
            X = self.resample_slices(X)

        return X, y


class NumpyDataset(data.Dataset):

    def __init__(self,
                 inputs,
                 labels,
                 input_size=(48, 224, 160),
                 sequences=['FLAIR', 'T1w', 'T1wCE', 'T2w'],
                 rescale=False,
                 add_noise=0.01,
                 flip=True,
                 tumor_only=False,
                 verbose=True,
                 start=None,
                 stop=None,
                 divisor=1,
                 test_mode=False):
        self.inputs = inputs
        self.labels = labels
        self.input_size = input_size
        self.sequences = sequences
        self.rescale = rescale
        self.add_noise = add_noise
        self.flip = flip
        self.tumor_only = tumor_only
        self.verbose = verbose
        self.start = start
        self.stop = stop
        self.divisor = divisor
        self.test_mode = test_mode
        self.augment = augmentations()

    def __len__(self): return len(self.inputs)

    @staticmethod
    def flip_array(X):
        # X.shape = (C, Z, H, W)
        if random.random() > 0.5:
            X = X[:, :, :, ::-1]
        if random.random() > 0.5:
            X = X[:, :, ::-1, :]
        if random.random() > 0.5:
            X = X[:, ::-1, :, :]
        X = np.ascontiguousarray(X)
        return X

    @staticmethod
    def get_pad_dims(actual, desired):
        diff = desired - actual
        pad1 = diff // 2
        pad2 = diff - pad1
        pad1 = max(0, pad1)
        pad2 = max(0, pad2)
        return (pad1, pad2)

    def pad_array(self, array, size):
        # array.shape = (C, Z, H, W)
        y_pad = self.get_pad_dims(array.shape[2], size[1])
        z_pad = self.get_pad_dims(array.shape[3], size[2])
        sequence_list = []
        for seq in range(array.shape[0]):
            sequence = np.pad(array[seq], 
                              [(0,0), y_pad, z_pad], 
                              mode='constant', 
                              constant_values=np.min(array[seq]))
            sequence_list.append(sequence)
        return np.stack(sequence_list)

    def pad_and_resize(self, X):
        # When resampled to 3x1x1, the max dimensions are for training set images are:
        #     53 x 210 x 155
        # So we can resize to 48 x 224 x 160
        # If <48 slices, pad to 48
        z, h, w = X.shape[1:]
        if z < self.input_size[0]:
            filler = np.expand_dims(np.zeros_like(X[:,0]), axis=1)
            for channel in range(filler.shape[0]):
                filler[channel][...] = np.min(X[channel])
            filler = np.repeat(filler, self.input_size[0] - z, axis=1)
            X = np.concatenate([filler, X], axis=1)
        # If >48 slices, this will get resampled after X 
        # is converted to torch Tensor
        X = self.pad_array(X, self.input_size)
        return X

    def resample_slices(self, X):
        X = F.interpolate(X.unsqueeze(0), size=tuple(self.input_size), mode='nearest')
        return X.squeeze(0)

    def get(self, i):   
        # try:
            X = np.stack([np.load(f'{osp.join(self.inputs[i], seq+".npy")}') for seq in self.sequences])
            if not isinstance(self.start, NONETYPE):
                start = self.start[i] // self.divisor
                stop = self.stop[i] // self.divisor
                X = X[:,start:stop+1]
            return X
        # except Exception as e:
        #     if self.verbose: print(e)
        #     return None

    def get_batch_and_labels(self, i):
        X = self.get(i)
        while isinstance(X, NONETYPE):
            i = np.random.randint(len(self))
            X = self.get(i)

        X = self.pad_and_resize(X)

        # if not self.test_mode:
        #     c, z, h, w = X.shape
        #     X = X.reshape(c*z, h, w)
        #     to_augment = {'image': X[0]}
        #     for i in range(1, len(X)):
        #         to_augment[f'image{i}'] = X[i]
        #     augmented = self.augment(**to_augment)
        #     X = np.concatenate([augmented['image'] if i == 0 else augmented[f'image{i}'] for i in range(len(X))])
        #     X = X.reshape(c, z, h, w)

        if self.add_noise > 0 and not self.test_mode:
            noise = np.random.normal(0, self.add_noise, X.shape)
            X[X > np.min(X)] += noise[X > np.min(X)]

        if self.rescale:
            X = X - np.min(X)
            X = X / np.max(X)

        if self.flip and not self.test_mode:
            X = self.flip_array(X)

        X = np.ascontiguousarray(X)
        X = torch.from_numpy(X).float()
        y = torch.tensor(self.labels[i])

        if X.shape[1] > self.input_size[0]:
            X = self.resample_slices(X)

        return X, y

    def __getitem__(self, i):
        return self.get_batch_and_labels(i)


class SiameseDataset(NumpyDataset):

    def __getitem__(self, i):
        X1, y1 = self.get_batch_and_labels(i)
        j = np.random.randint(len(self))
        while j == i:
            j = np.random.randint(len(self))
        X2, y2 = self.get_batch_and_labels(j)
        return (X1, X2), int(y1 == y2)


class NumpySegDataset(NumpyDataset):

    @staticmethod
    def flip_array(X, y):
        # X.shape = (C, Z, H, W)
        if random.random() > 0.5:
            X = X[:, :, :, ::-1]
            y = y[:, :, :, ::-1]
        if random.random() > 0.5:
            X = X[:, :, ::-1, :]
            y = y[:, :, ::-1, :]
        if random.random() > 0.5:
            X = X[:, ::-1, :, :]
            y = y[:, ::-1, :, :]
        if random.random() > 0.5 and X.shape[-1] == X.shape[-2]:
            X = X.transpose(0, 1, 3, 2)
            y = y.transpose(0, 1, 3, 2)
        X = np.ascontiguousarray(X)
        y = np.ascontiguousarray(y)
        return X, y


    def get(self, i):   
        # try:
            X = np.stack([np.load(f'{osp.join(self.inputs[i], seq+".npy")}') for seq in self.sequences])
            seg = np.expand_dims(np.load(osp.join(self.inputs[i], 'seg.npy')), axis=0)
            y = np.zeros_like(seg)
            y = np.repeat(y, 3, axis=0)
            #1- NECROSIS, 2- EDEMA, 4- ENHANCING
            y[0] = (seg == 4).astype('float') # Enhancing tumor
            y[1] = (seg > 0).astype('float') - (seg == 2).astype('float') # Tumor core
            y[2] = (seg > 0).astype('float') # Whole tumor
            return X, y
        # except Exception as e:
        #     if self.verbose: print(e)
        #     return None

    def __getitem__(self, i):
        data = self.get(i)
        while isinstance(data, NONETYPE):
            i = np.random.uniform(len(self))
            data = self.get(i)

        X, y = data

        X = self.pad_and_resize(X)
        y = self.pad_and_resize(y)

        if self.add_noise > 0 and not self.test_mode:
            noise = np.random.normal(0, self.add_noise, X.shape)
            X[X > np.min(X)] += noise[X > np.min(X)]

        if self.flip and not self.test_mode:
            X, y = self.flip_array(X, y)

        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()

        if X.shape[1] > self.input_size[0]:
            X = self.resample_slices(X)
            y = self.resample_slices(y)

        return X, y


class NumpySegDataset2(NumpySegDataset):

    def __init__(self, *args, **kwargs):
        self.class_label_as_seg = kwargs.pop('class_label_as_seg', False)
        super().__init__(*args, **kwargs)

    def get(self, i):   
        #try:
            X = np.stack([np.load(f'{osp.join(self.inputs[i], seq+".npy")}') for seq in self.sequences])
            seg = np.expand_dims(np.load(osp.join(self.inputs[i], 'SEG.npy')), axis=0)
            y = np.zeros_like(seg)
            if self.class_label_as_seg:
                if self.labels[i] == 1:
                    y[0] = (seg > 0).astype('float')
            else:
                y = np.repeat(y, 2, axis=0)
                #1- NECROSIS, 2- EDEMA, 4- ENHANCING
                #y[0] = (seg == 4).astype('float') # Enhancing tumor
                y[0] = (seg > 0).astype('float') - (seg == 2).astype('float') # Tumor core
                y[1] = (seg > 0).astype('float') # Whole tumor
            return X, y
        # except Exception as e:
        #     if self.verbose: print(e)
        #     return None


    def __getitem__(self, i):
        data = self.get(i)
        while isinstance(data, NONETYPE):
            i = np.random.uniform(len(self))
            data = self.get(i)

        X, y = data

        X = self.pad_and_resize(X)
        y = self.pad_and_resize(y)

        if self.add_noise > 0 and not self.test_mode:
            noise = np.random.normal(0, self.add_noise, X.shape)
            X[X > np.min(X)] += noise[X > np.min(X)]

        if self.flip and not self.test_mode:
            X, y = self.flip_array(X, y)

        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()

        if X.shape[1] > self.input_size[0]:
            X = self.resample_slices(X)
            y = self.resample_slices(y)

        return X, (self.labels[i], y)


