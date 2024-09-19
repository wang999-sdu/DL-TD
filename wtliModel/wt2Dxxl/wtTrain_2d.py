# 导入相应的库
import argparse
import math
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from tqdm import tqdm

from dataSet_2d import read_split_data, data_load
from dataloader_2d import My_Dataset
from utils_2d import matplotlib_imshow, train_one_epoch, evaluate
from torchsummary import summary


# 主函数
def main(opt):
    # 1.读取一些配置参数，并且输出
    print(opt)
    assert os.path.exists(opt.data_path), "{} dose not exists.".format(opt.data_path)

    # 创建日志文件
    tb_writer = SummaryWriter("./summaryLog")
    # 日志保存路径
    save_dir = tb_writer.log_dir
    # 模型保存路径
    weights_dir = save_dir + "/weights"
    # 如果文件夹不存在，则创建文件夹
    os.makedirs(weights_dir, exist_ok=True)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() and opt.use_cuda else "cpu")

    train_scans_np, train_y_set_np, test_scans_np, test_y_set_np = data_load()

    data_transform = {
        "train": transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    train_dataset = My_Dataset(image=train_scans_np, label=train_y_set_np, transform=data_transform['train'])
    val_dataset = My_Dataset(image=test_scans_np, label=test_y_set_np, transform=data_transform['val'])

    nw = min([os.cpu_count(), opt.batch_size, opt.num_worker if opt.batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
                                               num_workers=nw, collate_fn=train_dataset.collate_fn, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False,
                                             num_workers=nw, collate_fn=val_dataset.collate_fn, pin_memory=True)

    #   3.2 网络搭建：model
    model = resnet50()
    if opt.weights != '':
        assert os.path.exists(opt.weights), "weights file: '{}' not exist.".format(opt.weights)
        weights_dict = torch.load(opt.weights)
        in_channel = model.fc.in_features
        model.fc = nn.Linear(in_channel, 1)
        del_keys = ['fc.weight', 'fc.bias']
        for k in del_keys:
            del weights_dict[k]
        model.load_state_dict(weights_dict, strict=False)

    if opt.freeze_layers:
        for name, para in model.named_parameters():
            if "fc" not in name:
                para.requires_grad(False)
            else:
                print("training {}".format(name))
    model = model.to(device)

    #   3.3 优化器，学习率，更新策略,损失函数
    pg = [p for p in model.parameters() if p.requires_grad]
    if opt.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(pg, lr=opt.lr, momentum=0.9, weight_decay=5e-5)
    else:
        optimizer = torch.optim.Adam(pg, lr=opt.lr, weight_decay=1e-3)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / opt.epochs)) / 2) * (1 - opt.lrf) + opt.lrf  # cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], 0.1)
    # loss_function = torch.nn.CrossEntropyLoss().to(device)
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4])).to(device)
    #   3.4 模型训练：train
    best_acc = -np.inf
    best_epoch = 0

    # summary(model, (3, 7, 7))  # 输出网络结构
    for epoch in tqdm(range(opt.epochs)):
        # train
        train_loss, train_acc = train_one_epoch(model, train_loader, device, optimizer, loss_function, epoch=epoch)
        print(f'Epoch: {epoch}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}')
        scheduler.step()
        #  eval
        val_loss, val_acc = evaluate(model, val_loader, device, loss_function, epoch)
        print(f'Epoch: {epoch}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}')
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate", "images"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        batch_images = next(iter(train_loader))[0]
        tb_writer.add_images(tags[5], batch_images, epoch)

        #   3.6 模型保存：save
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_epoch = epoch + 1
        model_path = weights_dir + "/model_{}.pth".format(epoch)
        torch.save(model.state_dict(), model_path)
        if is_best:
            shutil.copy(model_path, weights_dir + "/best_model.pth")
    tb_writer.close()


# 程序入口
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default=r'G:\data-CT\png224',
                        help='The data path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-worker', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # resnet50预训练权重
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    parser.add_argument('--weights', type=str,
                        default=r'E:\workspace_python311\project-thymoma\wtlizzz-core\model\trails\models\resnet50'
                                r'-19c8e357.pth',
                        help='initial weights path')  #
    parser.add_argument('--freeze-layers', type=bool, default=False)
    # parser.add_argument('--use_cuda', default=False, type=bool)
    parser.add_argument('--use_cuda', default=True)
    parser.add_argument('--optimizer', type=str, default='SGD')  # Adam

    args = parser.parse_args()

    main(opt=args)
