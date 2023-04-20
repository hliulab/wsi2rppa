#!/usr/local/bin/python3
# -*- coding: utf-8 -*-


import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

# inner import
from datasets.tumor_dataset import TumorDataset
from models.clam_model import CLAM_SB


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
batch_size = 1
parser = argparse.ArgumentParser(description='Tumor Classification Training')

parser.add_argument('--csv_path', default='')
parser.add_argument('--lr', default=1e-4)
parser.add_argument('--wsi_path', default='')
parser.add_argument('--train_label_path', default='')
parser.add_argument('--val_label_path', default='')
parser.add_argument('--lr', default=1e-4)
parser.add_argument('--epochs', default=30)
args = parser.parse_args()
date = datetime.now().strftime(r'%m%d_%H%M%S')

train_set = TumorDataset(path=args.wsi_path,
                         csv_path=args.train_label_path)
valid_set = TumorDataset(path=args.wsi_path,
                         csv_path=args.val_label_path)

train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=1, shuffle=True)
val_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=1, shuffle=True)

model = CLAM_SB(n_classes=2, dropout=False, dropout_ratio=0.2, size_arg='tumor')


def train(arg, checkpoint_path: str = None) -> None:
    model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=arg.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        del checkpoint['classifiers.weight'], checkpoint['classifiers.bias']
        model.load_state_dict(checkpoint, strict=False)
        start_epoch = 1
        print('find model')
    else:
        start_epoch = 1
        print('no model')

    result_train_loss = []
    result_valid_loss = []
    result_train_acc = []
    result_valid_acc = []
    result_train_auc = []
    result_valid_auc = []

    if not os.path.exists('./model_result_tumor'):
        os.mkdir('./model_result_tumor')
    print("Begin training...")
    for epoch in range(start_epoch, arg.epochs + 1):
        model.train()
        running_train_loss = 0.0
        running_val_loss = 0.0
        # Training Loop
        train_correct = 0
        train_total = 0
        valid_correct = 0
        valid_total = 0
        y_prob_list_train = []
        y_prob_list_valid = []
        labels_list_train = []
        labels_list_valid = []
        labels_list_valid_predicted = []
        for data in train_loader:
            train_total += 1
            wsi_feature, label = data  # 返回WSI特征和标签
            optimizer.zero_grad()  # zero the parameter gradients
            wsi_feature = torch.squeeze(wsi_feature)
            wsi_feature = wsi_feature.to(device)
            logits, y_prob, *_ = model(wsi_feature)  # predict output from the model
            logits = logits.to(device)
            _, predicted = torch.max(y_prob, 1)
            label = label.to(device)
            one_hot_labels = F.one_hot(label, num_classes=2)
            y_prob_list_train.extend(y_prob[:, 1].detach().cpu().numpy())
            labels_list_train.extend(label.detach().cpu().numpy())

            train_correct += (predicted == label).sum()
            train_loss = loss_fn(logits, one_hot_labels.float())  # calculate loss for the predicted output
            train_loss.backward()  # back propagate the loss
            optimizer.step()  # adjust parameters based on the calculated gradients
            running_train_loss += train_loss.item()  # track the loss value
        # Calculate training loss value
        fpr, tpr, _ = metrics.roc_curve(labels_list_train, y_prob_list_train)
        train_auc = metrics.auc(fpr, tpr)
        train_loss_value = running_train_loss / len(train_loader)

        scheduler.step()

        # Validation Loop
        with torch.no_grad():
            model.eval()
            for data in val_loader:
                valid_total += 1
                wsi_feature, label = data
                wsi_feature = torch.squeeze(wsi_feature)
                wsi_feature = wsi_feature.to(device)
                logits, y_prob, *_ = model(wsi_feature)
                _, predicted = torch.max(y_prob, 1)
                label = label.to(device)
                one_hot_labels = F.one_hot(label, num_classes=2)
                logits = logits.to(device)
                val_loss = loss_fn(logits, one_hot_labels.float())
                y_prob_list_valid.extend(y_prob[:, 1].detach().cpu().numpy())
                labels_list_valid.extend(label.detach().cpu().numpy())
                labels_list_valid_predicted.extend(predicted.detach().cpu().numpy())
                valid_correct += (predicted == label).sum()
                # The label with the highest value will be our prediction
                running_val_loss += val_loss.item()

        # Calculate validation loss value
        val_loss_value = running_val_loss / len(val_loader)
        fpr, tpr, _ = metrics.roc_curve(labels_list_valid, y_prob_list_valid)
        precision, recall, _ = metrics.precision_recall_curve(labels_list_valid, y_prob_list_valid)
        valid_auc = metrics.auc(fpr, tpr)
        # Calculate accuracy as the number of correct predictions in the validation batch divided by the total_val
        # number of predictions done.
        # Save the model if the accuracy is the best

        # Print the statistics of the epoch
        print('Completed training epoch', epoch, 'Training Loss is: %.4f' % train_loss_value,
              'Training Acc is : %.4f' % (train_correct / train_total),
              'Training Auc is : %.4f' % train_auc,
              'Validation Loss is: %.4f' % val_loss_value,
              'Valid Acc is : %.4f' % (valid_correct / valid_total),
              'Valid Auc is : %.4f' % valid_auc)

        result_train_loss.append(float(train_loss_value))
        result_valid_loss.append(float(val_loss_value))
        result_train_acc.append(float(train_correct / train_total))
        result_valid_acc.append(float(valid_correct / valid_total))
        result_train_auc.append(float(train_auc))
        result_valid_auc.append(float(valid_auc))

        # early_stopping(epoch, val_loss_value, model, optimizer, ckpt_name='/data/data_xxd/ckpt/yale_her2+-.pt')
        torch.save(model.state_dict(), 'xxx.pt')
    df = pd.DataFrame({
        'train_loss': result_train_loss,
        'valid_loss': result_valid_loss,
        'train_acc': result_train_acc,
        'valid_acc': result_valid_acc,
        'train_auc': result_train_auc,
        'valid_auc': result_valid_auc,
        'info': f'epoch={arg.epochs},optimizer={optimizer.defaults}',
    }, index=range(1, len(result_train_loss) + 1))
    df.to_csv(f"./model_result_tumor/result_tumor_{date}.csv")


if __name__ == "__main__":
    train(args)
    pass
