import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

root_gen = r''


def read_h5file(path):
    with h5py.File(path, 'r') as hdf5_file:
        features = hdf5_file['features'][:]
    return features


def normalized(data):
    data = np.array(data)
    scale = MinMaxScaler(feature_range=(0, 1))
    return np.squeeze(scale.fit_transform(data.reshape((-1, 1))))


class FeatureDataset(Dataset):
    def __init__(self, feature_path=None, data_path=None):
        super(FeatureDataset, self).__init__()
        self.path = feature_path
        self.data_path = data_path
        self.feature = []
        df = pd.read_csv(data_path, sep=',', header=0)
        all_protein = df.columns.tolist()
        for idx, row in df.iterrows():
            path = row['path']
            self.feature.append(
                (self.path + str(path),
                 list(row[all_protein.index('X1433EPSILON'):all_protein.index('PDL1') + 1])))

    def __getitem__(self, item) -> tuple:
        feature_h5path, data = self.feature[item]
        with h5py.File(feature_h5path, 'r') as hdf5_file:
            features = hdf5_file['features'][:]
        features = torch.from_numpy(features)
        data = torch.as_tensor(data)
        return features, data

    def __len__(self) -> int:
        return len(self.feature)
