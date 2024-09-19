import numpy as np
import torch
import time
import os
from torch import nn
from scipy import ndimage
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from tqdm import tqdm
from utils.logger import log
from wtliModel.wtModel.wtModel import generate_model


def wtTrain(data_loader_train, data_loader_val, summaryWriter, model, optimizer, scheduler, criterion, total_epochs, save_interval,
            save_folder,
            sets):
    # settings
    batches_per_epoch = len(data_loader_train)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    # device = torch.device("cpu")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for epoch in range(total_epochs):
        start = time.time()
        per_epoch_loss = 0
        num_correct = 0
        score_list = []
        label_list = []

        val_num_correct = 0
        val_score_list = []
        val_label_list = []

        model.train()
        with torch.enable_grad():
            for x, label in tqdm(data_loader_train):
                x = x.float()
                x = x.to(device)
                label = label.to(device)
                # label = torch.squeeze(label)
                label_list.extend(label.cpu().numpy())
                # print(label_list)

                # Forward pass
                logits = model(x)
                logits = logits.reshape([label.cpu().numpy().shape[0]])
                prob_out = nn.Sigmoid()(logits)
                # print(logits.shape)

                pro_list = prob_out.detach().cpu().numpy()
                for i in range(pro_list.shape[0]):
                    if (pro_list[i] > 0.5) == label.cpu().numpy()[i]:
                        num_correct += 1
                score_list.extend(pro_list)

                # label = torch.Tensor(label)
                logits = logits.to(device)
                print("---logits:{}----label:{}------", logits, label)

                loss = criterion(logits, label.float())

                per_epoch_loss += loss.item()
                print("---loss.item():{}------", loss.item())
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # pred = logits.argmax(dim=1)
                # num_correct += torch.eq(pred, label).sum().float().item()

            score_array = np.array(score_list)
            label_array = np.array(label_list)
            fpr_keras_1, tpr_keras_1, thresholds_keras_1 = roc_curve(label_array, score_array)
            auc_keras_1 = auc(fpr_keras_1, tpr_keras_1)

            print("Train EVpoch: {}\t Loss: {:.6f}\t Acc: {:.6f} AUC: {:.6f} ".format(epoch, per_epoch_loss / len(
                data_loader_train), num_correct / len(data_loader_train.dataset), auc_keras_1))
            summaryWriter.add_scalars('loss', {"loss": (per_epoch_loss / len(data_loader_train))}, epoch)
            summaryWriter.add_scalars('acc', {"acc": num_correct / len(data_loader_train.dataset)}, epoch)
            summaryWriter.add_scalars('auc', {"auc": auc_keras_1}, epoch)

        model.eval()
        with torch.no_grad():
            for x, label in tqdm(data_loader_val):
                x = x.float()
                x = x.to(device)
                label = label.to(device)

                val_label_list.extend(label.cpu().numpy())

                # Forward pass
                logits = model(x)
                logits = logits.reshape([label.cpu().numpy().shape[0]])
                prob_out = nn.Sigmoid()(logits)
                pro_list = prob_out.detach().cpu().numpy()

                for i in range(pro_list.shape[0]):
                    if (pro_list[i] > 0.5) == label.cpu().numpy()[i]:
                        val_num_correct += 1

                val_score_list.extend(pro_list)

            score_array = np.array(val_score_list)
            label_array = np.array(val_label_list)
            fpr_keras_1, tpr_keras_1, thresholds_keras_1 = roc_curve(label_array, score_array)
            auc_keras_1 = auc(fpr_keras_1, tpr_keras_1)

            print(
                "val Epoch: {}\t Acc: {:.6f} AUC: {:.6f} ".format(epoch, val_num_correct / len(data_loader_val.dataset),
                                                                  auc_keras_1))
            summaryWriter.add_scalars('acc', {"val_acc": val_num_correct / len(data_loader_val.dataset)}, epoch)
            summaryWriter.add_scalars('auc', {"val_auc": auc_keras_1}, epoch)
            summaryWriter.add_scalars('time', {"time": (time.time() - start)}, epoch)

        scheduler.step()

        filepath = "./wtWeights"
        folder = os.path.exists(filepath)
        if not folder:
            # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(filepath)
        path = './wtWeights/model' + str(epoch) + '.pth'
        torch.save(model.state_dict(), path)
