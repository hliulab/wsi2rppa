# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import argparse
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from datetime import datetime
from datasets.feature_dataset import FeatureDataset
from models.att_model import CLAM_SB_Reg
# inner import
from utils.early_stopping_utils import EarlyStopping

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1
date = datetime.now().strftime(r'%m%d_%H%M%S')


def train(num_epochs: int, model, train_loader, val_loader, checkpoint_path: str = None) -> None:
    model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    loss_f1 = nn.L1Loss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        del checkpoint['classifiers.weight'], checkpoint['classifiers.bias']
        model.load_state_dict(checkpoint, strict=False)
        start_epoch = 1
        print('no model')
    else:
        start_epoch = 1
        print('find model')

    result_train_loss = []
    result_valid_loss = []

    if not os.path.exists('./model_result'):
        os.mkdir('./model_result')
    early_stopping = EarlyStopping(patience=3, stop_epoch=10, verbose=False)
    print("Begin training...")
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        running_train_loss = 0.0
        running_val_loss = 0.0
        running_train_L1_loss = 0.0
        running_val_L1_loss = 0.0
        # Training Loop
        for data in train_loader:
            wsi_feature, protein_data = data
            optimizer.zero_grad()  # zero the parameter gradients
            wsi_feature = torch.squeeze(wsi_feature)
            wsi_feature = wsi_feature.to(device)
            predicted_outputs = model(wsi_feature)  # predict output from the model
            predicted_outputs = predicted_outputs.to(device)
            protein_data = protein_data.to(device)
            train_loss = loss_fn(predicted_outputs, protein_data)  # calculate loss for the predicted output
            L1_loss = loss_f1(predicted_outputs, protein_data)
            train_loss.backward()  # back propagate the loss
            optimizer.step()  # adjust parameters based on the calculated gradients
            running_train_loss += train_loss.item()  # track the loss value
            running_train_L1_loss += L1_loss.item()
        # Calculate training loss value
        train_loss_value = running_train_loss / len(train_loader)
        running_train_L1_loss = running_train_L1_loss / len(train_loader)

        scheduler.step()

        # Validation Loop
        with torch.no_grad():
            model.eval()
            for data in val_loader:
                wsi_feature, protein_data = data
                wsi_feature = torch.squeeze(wsi_feature)
                wsi_feature = wsi_feature.to(device)
                predicted_outputs = model(wsi_feature)
                protein_data = protein_data.to(device)
                predicted_outputs = predicted_outputs.to(device)
                val_loss = loss_fn(predicted_outputs, protein_data)
                L1_loss = loss_f1(predicted_outputs, protein_data)
                # The label with the highest value will be our prediction
                running_val_L1_loss += L1_loss.item()
                running_val_loss += val_loss.item()

        # Calculate validation loss value
        val_loss_value = running_val_loss / len(val_loader)
        running_val_L1_loss = running_val_L1_loss / len(val_loader)
        # Calculate accuracy as the number of correct predictions in the validation batch divided by the total_val
        # number of predictions done.
        # Save the model if the accuracy is the best

        # Print the statistics of the epoch
        print('Completed training epoch', epoch, 'Training Loss is: %.4f' % train_loss_value,
              'Train L1Loss is : %.4f' % running_train_L1_loss,
              'Validation Loss is: %.4f' % val_loss_value, 'Val L1Loss is : %.4f' % running_val_L1_loss)

        result_train_loss.append(train_loss_value)
        result_valid_loss.append(val_loss_value)

        early_stopping(epoch, val_loss_value, model, optimizer,
                       ckpt_name=f'./model_result/checkpoint_{args.suffix}_{date}.pt')
        if early_stopping.early_stop:
            print("stop train")
            break

    plt.plot(range(1, len(result_train_loss) + 1), result_train_loss, label='train loss', color='r')
    plt.plot(range(1, len(result_valid_loss) + 1), result_valid_loss, label='valid loss', color='g')
    plt.legend()
    plt.savefig(f'./model_result/{args.suffix}_{date}.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Protein level Training')
    parser.add_argument('--suffix', default='')
    parser.add_argument('--feature_path', type=str, metavar='DIR', help='path to feature')
    parser.add_argument('--train_csv_path')
    parser.add_argument('--val_csv_path')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--num_workers')
    args = parser.parse_args()

    model = CLAM_SB_Reg(n_classes=223)
    train_set = FeatureDataset(feature_path=args.feature_path, data_path=args.train_csv_path)
    valid_set = FeatureDataset(feature_path=args.feature_path, data_path=args.val_csv_path)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=args.num_workers, shuffle=True)
    train(args.epoch, model, train_loader, val_loader)
