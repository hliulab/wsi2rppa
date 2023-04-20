# !/usr/bin/env python
# -*- coding:utf-8 -*-
import os

import h5py
import numpy as np

from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import isWhitePatch, isBlackPatch


def save_patch(h5_file_path, wsi_path, save_dir):
    wsi = WholeSlideImage(wsi_path)
    os.makedirs(save_dir, exist_ok=True)
    with h5py.File(h5_file_path, "r") as f:
        dset = f['coords']
        patch_size = f['coords'].attrs['patch_size']
        patch_level = f['coords'].attrs['patch_level']
        for idx, c in enumerate(dset):
            tile = wsi.wsi.read_region(c, patch_level, (patch_size, patch_size)).convert('RGB')
            if isWhitePatch(np.array(tile), 15) or isBlackPatch(np.array(tile), 50):
                print('continue', idx)
                continue
            tile.save(os.path.join(save_dir, str(idx) + '.png'))


if __name__ == "__main__":
    pass
