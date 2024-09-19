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
from TDModel.TDModel.wtModel import generate_model


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=5, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, name="checkpoint.pt"):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, name)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, name):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), name)
        self.val_loss_min = val_loss


def wtTrain(
    data_loader_train,
    # data_loader_test,
    summaryWriter,
    model,
    optimizer,
    scheduler,
    criterion,
    total_epochs,
    save_interval,
    save_folder,
    sets,
):
    # settings
    batches_per_epoch = len(data_loader_train)
    log.info(
        "{} epochs in total, {} batches per epoch".format(
            total_epochs, batches_per_epoch
        )
    )
    # device = torch.device("cpu")
    print(f"==sets.gpu_id = {sets.gpu_id}")
    if torch.cuda.is_available():
        device = torch.device("cuda", sets.gpu_id)
    else:
        device = torch.device("cpu")
    print(f"device = {device}")
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
                # logits = model(x)
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
                print("---epoch:{}----", epoch)
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
            fpr_keras_1, tpr_keras_1, thresholds_keras_1 = roc_curve(
                label_array, score_array
            )
            auc_keras_1 = auc(fpr_keras_1, tpr_keras_1)

            print(
                "Train EVpoch: {}\t Loss: {:.6f}\t Acc: {:.6f} AUC: {:.6f} ".format(
                    epoch,
                    per_epoch_loss / len(data_loader_train),
                    num_correct / len(data_loader_train.dataset),
                    auc_keras_1,
                )
            )
            summaryWriter.add_scalars(
                "loss", {"loss": (per_epoch_loss / len(data_loader_train))}, epoch
            )
            summaryWriter.add_scalars(
                "acc", {"acc": num_correct / len(data_loader_train.dataset)}, epoch
            )
            summaryWriter.add_scalars("auc", {"auc": auc_keras_1}, epoch)

        model.eval()
        # with torch.no_grad():
        #     for x, label in tqdm(data_loader_test):
        #         x = x.float()
        #         x = x.to(device)
        #         label = label.to(device)

        #         val_label_list.extend(label.cpu().numpy())

        #         # Forward pass
        #         logits = model(x)
        #         logits = logits.reshape([label.cpu().numpy().shape[0]])
        #         prob_out = nn.Sigmoid()(logits)
        #         pro_list = prob_out.detach().cpu().numpy()

        #         for i in range(pro_list.shape[0]):
        #             if (pro_list[i] > 0.5) == label.cpu().numpy()[i]:
        #                 val_num_correct += 1

        #         val_score_list.extend(pro_list)

        #     score_array = np.array(val_score_list)
        #     label_array = np.array(val_label_list)
        #     fpr_keras_1, tpr_keras_1, thresholds_keras_1 = roc_curve(
        #         label_array, score_array
        #     )
        #     auc_keras_1 = auc(fpr_keras_1, tpr_keras_1)

        #     print(
        #         "val Epoch: {}\t Acc: {:.6f} AUC: {:.6f} ".format(
        #             epoch, val_num_correct / len(data_loader_test.dataset), auc_keras_1
        #         )
        #     )
        #     summaryWriter.add_scalars(
        #         "acc",
        #         {"val_acc": val_num_correct / len(data_loader_test.dataset)},
        #         epoch,
        #     )
        #     summaryWriter.add_scalars("auc", {"val_auc": auc_keras_1}, epoch)
        #     summaryWriter.add_scalars("time", {"time": (time.time() - start)}, epoch)

        scheduler.step()
        if (epoch + 1) % save_interval == 0:
            path = (
                save_folder
                # "/home/wangwenmiao/wtli-workspace/project-thymoma/wtlizzz-core/wtliModel/wtWeight/model"
                + str(epoch + 1)
                + ".pth"
            )
            torch.save(model.state_dict(), path)
