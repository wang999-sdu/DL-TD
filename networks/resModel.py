from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import time
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torchsummary
import time
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter

from networks.component.dataLoadMethod import download_data
from networks.component.showPic import show_pic
from networks.modelDir import generate_model

summaryWriter = SummaryWriter("./logs/")
batch_size = 256


def data_loader(train_dataset, val_dataset, test_dataset):
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False)
    for x, y in train_loader:
        print(x.shape, y.shape)
        break
    # torch.Size([256, 1, 28, 28, 28]) torch.Size([256, 1])
    return train_loader, val_loader, test_loader


def res_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("cpu")
    print('device =', device)
    print(torch.cuda.get_device_name(0))
    model = generate_model(model_type='resnet', model_depth=50,
                           input_W=224, input_H=224, input_D=224, resnet_shortcut='B',
                           no_cuda=False, gpu_id=[0],
                           pretrain_path='../model/pretrain/resnet_50_23dataset.pth',
                           nb_class=11)
    return device, model


def train_model(device, train_loader, val_loader, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # criterion = nn.CrossEntropyLoss()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0])).cuda()  # 分类不均衡
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    num_epochs = 200
    total_step = len(train_loader)
    time_list = []
    for epoch in range(num_epochs):
        start = time.time()
        per_epoch_loss = 0
        num_correct = 0
        val_num_correct = 0
        model.train()
        with torch.enable_grad():
            for x, label in tqdm(train_loader):
                x = x.to(device)
                label = label.to(device)
                label = torch.squeeze(label)  # label的形状是 [256,1] 要将其变成 [256]
                # Forward pass
                logits = model(x)
                loss = criterion(logits, label.long())
                per_epoch_loss += loss.item()
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pred = logits.argmax(dim=1)
                num_correct += torch.eq(pred, label).sum().float().item()
            print("Train Epoch: {}\t Loss: {:.6f}\t Acc: {:.6f}".format(epoch, per_epoch_loss / total_step,
                                                                        num_correct / len(train_loader.dataset)))
            summaryWriter.add_scalars('loss', {"loss": (per_epoch_loss / total_step)}, epoch)
            summaryWriter.add_scalars('acc', {"acc": num_correct / len(train_loader.dataset)}, epoch)
        model.eval()
        with torch.no_grad():
            for x, label in tqdm(val_loader):
                x = x.to(device)
                label = label.to(device)
                label = torch.squeeze(label)
                # Forward pass
                logits = model(x)
                pred = logits.argmax(dim=1)
                val_num_correct += torch.eq(pred, label).sum().float().item()
            print("val Epoch: {}\t Acc: {:.6f}".format(epoch, num_correct / len(train_loader.dataset)))
            summaryWriter.add_scalars('acc', {"val_acc": val_num_correct / len(val_loader.dataset)}, epoch)
            summaryWriter.add_scalars('time', {"time": (time.time() - start)}, epoch)
        scheduler.step()
    torch.save(model, './wtlizzzResModel.pth')


def init_res_model():
    train_dataset, val_dataset, test_dataset = download_data()
    train_loader, val_loader, test_loader = data_loader(train_dataset, val_dataset, test_dataset)
    device, model = res_model()
    train_model(device, train_loader, val_loader, model)


if __name__ == '__main__':
    init_res_model()
