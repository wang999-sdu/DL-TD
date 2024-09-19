from torch.optim.lr_scheduler import ExponentialLR

from networks.modelDir import generate_model
import torch
import torch.nn as nn


def get_model():
    model = generate_model(model_type='resnet', model_depth=50,
                           input_W=224, input_H=224, input_D=224, resnet_shortcut='B',
                           no_cuda=False, gpu_id=[0],
                           pretrain_path='../model/pretrain/resnet_50_23dataset.pth',
                           nb_class=11)

def train_model(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    num_epochs = 800
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
                loss = criterion(logits, label)

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


