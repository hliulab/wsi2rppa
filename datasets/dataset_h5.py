from __future__ import print_function, division

from abc import ABC

import numpy as np
import torch
import h5py
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def eval_transforms(pretrained=False):
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    transforms_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    )

    return transforms_val


class Whole_Slide_Bag(Dataset):
    def __init__(self,
                 file_path,
                 pretrained=False,
                 custom_transforms=None,
                 target_patch_size=-1,
                 ):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
        """
        self.pretrained = pretrained
        if target_patch_size > 0:
            self.target_patch_size = (target_patch_size, target_patch_size)
        else:
            self.target_patch_size = None

        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
        else:
            self.roi_transforms = custom_transforms

        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f['imgs']
            self.length = len(dset)

        self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['imgs']
        for name, value in dset.attrs.items():
            print(name, value)

        print('pretrained:', self.pretrained)
        print('transformations:', self.roi_transforms)
        if self.target_patch_size is not None:
            print('target_size: ', self.target_patch_size)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as hdf5_file:
            img = hdf5_file['imgs'][idx]
            coord = hdf5_file['coords'][idx]

        img = Image.fromarray(img)
        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
        img = self.roi_transforms(img).unsqueeze(0)
        return img, coord


class Whole_Slide_Bag_FP(Dataset):
    def __init__(self,
                 file_path,
                 wsi,
                 simclr_transforms=None,
                 train_simclr=False,
                 pretrained=False,
                 custom_transforms=None,
                 custom_downsample=1,
                 target_patch_size=-1,
                 bag_name=None
                 ):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
            custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
            target_patch_size (int): Custom defined image size before embedding
        """
        self.pretrained = pretrained
        self.wsi = wsi
        self.simclr_transforms = simclr_transforms
        self.train_simclr = train_simclr
        self.bag_name = bag_name
        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
        else:
            self.roi_transforms = custom_transforms

        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f['coords']
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            self.length = len(dset)
            if target_patch_size > 0:
                self.target_patch_size = (target_patch_size,) * 2
            elif custom_downsample > 1:
                self.target_patch_size = (self.patch_size // custom_downsample,) * 2
            else:
                self.target_patch_size = None
        self.h5py_file = h5py.File(self.file_path, 'r')
        self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['coords']
        for name, value in dset.attrs.items():
            print(name, value)

        print('\nfeature extraction settings')
        print('target patch size: ', self.target_patch_size)
        print('pretrained: ', self.pretrained)
        print('transformations: ', self.roi_transforms)

    def __getitem__(self, idx):
        if self.train_simclr:
            all_coord = self.h5py_file['coords']
            coord = all_coord[idx]
            img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

            if self.target_patch_size is not None:
                img = img.resize(self.target_patch_size)

            if self.simclr_transforms is not None:
                imgL, imgR = self.simclr_transforms(img)
            return (imgL, imgR), coord
        else:
            if self.bag_name is None:
                coord = self.h5py_file['coords'][idx]
                img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

                if self.target_patch_size is not None:
                    img = img.resize(self.target_patch_size)
                img = self.roi_transforms(img).unsqueeze(0)

                return img, coord
            else:
                coord = self.random_coord[idx]
                img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

                if self.target_patch_size is not None:
                    img = img.resize(self.target_patch_size)
                img = self.roi_transforms(img).unsqueeze(0)
                protein = self.protein[self.protein['path'] == self.bag_name]
                return img, coord, torch.as_tensor(protein.loc[:, 'X1433EPSILON':'PDL1'].values, dtype=torch.float32)


class Dataset_All_Bags(Dataset):

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df['slide_id'][idx]


class DatasetBags(Dataset_All_Bags):

    def __init__(self, csv_path):
        super(DatasetBags, self).__init__(csv_path)

    def __getitem__(self, idx):
        return self.df['path'][idx]


class DatasetBagsOV(Dataset_All_Bags):

    def __init__(self, csv_path):
        super(DatasetBagsOV, self).__init__(csv_path)

    def __getitem__(self, idx):
        return self.df['filename'][idx], self.df['slide_path'][idx]


class RandomSampleFromWSI(Dataset, ABC):

    def __init__(self, file_path, wsi, pretrained=False, start_protein='X1433EPSILON',
                 end_protein='PDL1',
                 custom_transforms=None,
                 custom_down_sample=1,
                 target_patch_size=-1,
                 sample_percent=0.1,
                 tcpa_csv=None,
                 bag_name=None):
        self.pretrained = pretrained
        self.wsi = wsi
        self.bag_name = bag_name
        self.protein = pd.read_csv(tcpa_csv)
        self.start_protein = start_protein
        self.end_protein = end_protein
        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
        else:
            self.roi_transforms = custom_transforms

        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            if target_patch_size > 0:
                self.target_patch_size = (target_patch_size,) * 2
            elif custom_down_sample > 1:
                self.target_patch_size = (self.patch_size // custom_down_sample,) * 2
            else:
                self.target_patch_size = None
        self.h5py_file = h5py.File(self.file_path, 'r')
        coords_array = np.array(self.h5py_file['coords'])
        sample_num = int(coords_array.shape[0] * sample_percent)
        np.random.shuffle(coords_array)
        self.random_coord = coords_array[0:sample_num]
        self.length = sample_num

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        coord = self.random_coord[idx]
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
        img = self.roi_transforms(img)
        protein = self.protein[self.protein['path'] == self.bag_name]
        return img, torch.as_tensor(protein.loc[:, self.start_protein:self.end_protein].values, dtype=torch.float32)
