#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import h5py
import torch
import torchtuples as tt
import os
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from sklearn.decomposition import PCA


np.random.seed(1234)
_ = torch.manual_seed(123)
df = pd.read_csv('')

X_train = []
Y_train = []
for idx, row in df.iterrows():
    with h5py.File(os.path.join(row['path'])) as hdf5_file:
        features = hdf5_file['features'][:]
        X_train.append(features)
        Y_train.append([row['OS'], row['OS month']])

X_train = np.array(X_train)
X_train = X_train.squeeze()
Y_train = np.array(Y_train)

# pca
pca = PCA(n_components=10)
X_train = pca.fit_transform(X_train)
boundary = int(X_train.shape[0] * 0.7)

X_val = X_train[boundary:, :]
Y_val = Y_train[boundary:, :]
X_train = X_train[:boundary, :]
Y_train = Y_train[:boundary, :]
events_train = Y_train[:, 0]
durations_train = Y_train[:, 1]
events_val = Y_val[:, 0]
durations_val = Y_val[:, 1]

in_features = X_train.shape[1]
num_nodes = [256, 128]
out_features = 1
batch_norm = True
dropout = 0.1
output_bias = False
net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                              dropout, output_bias=output_bias)
model = CoxPH(net, tt.optim.SGD)
batch_size = 512
lrfinder = model.lr_finder(X_train, [durations_train, events_train], batch_size, tolerance=10)

model.optimizer.set_lr(0.01)
epochs = 512
callbacks = [tt.callbacks.EarlyStopping()]
verbose = True
log = model.fit(X_train, [durations_train, events_train], batch_size, epochs, callbacks, verbose,
                val_data=(X_val, [durations_val, events_val]), val_batch_size=batch_size)

model.partial_log_likelihood(*(X_val, [durations_val, events_val])).mean()

baseline_hazards = model.compute_baseline_hazards()
baseline_cumulative_hazards = model.compute_baseline_cumulative_hazards()

surv = model.predict_surv_df(X_val, baseline_hazards_=baseline_hazards)

ev = EvalSurv(surv, durations_val, events_val, censor_surv='km')
print('c-index=', ev.concordance_td())
time_grid = np.linspace(durations_val.min(), durations_val.max(), 100)



