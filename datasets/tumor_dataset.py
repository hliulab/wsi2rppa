#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""

@File    : response_dataset.py
@Author  : Dong
@Time    : 2022/9/18 21:26

"""
import os

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TumorDataset(Dataset):
    def __init__(self, path, csv_path=None, train_set_ratio=1, transforms_csv=None):
        super(TumorDataset, self).__init__()
        self.feature = []
        self.path = path
        df = pd.read_csv(csv_path, sep=',', header=0)

        for idx, row in df.iterrows():
            self.feature.append((os.path.join(self.path, row['path']), row['label']))

    def __getitem__(self, item) -> tuple:

        feature_h5path, label = self.feature[item]
        with h5py.File(feature_h5path, 'r') as hdf5_file:
            features = hdf5_file['features'][:]
        features = torch.from_numpy(features)
        return features, torch.tensor(label)

    def __len__(self) -> int:
        return len(self.feature)