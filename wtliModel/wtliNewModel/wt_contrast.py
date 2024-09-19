import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, average_precision_score, \
    roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.models import ResNet101_Weights, ResNet18_Weights, ResNet152_Weights, ResNet50_Weights
import pandas as pd
import os
from wtliModel.wtCommons.wt_earlys import wt_EarlyStopping
from wtliModel.wtliNewModel.data_process import CTDataset
import torchio as tio
from wtliModel.wtCommons.wt_log import Wt_logger_manager
from scipy.signal import savgol_filter
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from sklearn.svm import SVC
import timm
from torch_geometric.nn import GCNConv


class GNN_CT_Classifier(nn.Module):
    def __init__(self, cnn_feature_dim, gnn_hidden_dim, num_classes):
        super(GNN_CT_Classifier, self).__init__()
        self.gcn1 = GCNConv(cnn_feature_dim, gnn_hidden_dim)
        self.gcn2 = GCNConv(gnn_hidden_dim, gnn_hidden_dim)
        self.fc = nn.Linear(gnn_hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = nn.ReLU()(x)
        x = self.gcn2(x, edge_index)
        x = self.fc(x)
        return x


class VitClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(VitClassifier, self).__init__()
        # 加载预训练的 ViT 模型
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        # 修改分类头用于二分类
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18Classifier, self).__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet50Classifier, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


class ResNet101Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet101Classifier, self).__init__()
        self.resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

class ResNet152Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet152Classifier, self).__init__()
        self.resnet = models.resnet152(weights=ResNet152_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


class VggClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(VggClassifier, self).__init__()
        self.vgg = models.vgg19(pretrained=True)
        self.vgg.classifier[6] = nn.Linear(self.vgg.classifier[6].in_features, 2)

    def forward(self, x):
        return self.vgg(x)


class InceptionClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(InceptionClassifier, self).__init__()
        self.inception = models.inception_v3(pretrained=True)
        self.inception.BasicConv2d = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)  # 使用3x3的卷积核
        self.inception.fc = nn.Linear(self.inception.fc.in_features, 2)

    def forward(self, x):
        return self.inception(x)


class DensenetClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(DensenetClassifier, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, 2)

    def forward(self, x):
        return self.densenet(x)


class MobilenetClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(MobilenetClassifier, self).__init__()
        self.mobilenet = models.mobilenet_v3_large(pretrained=True)
        self.mobilenet.classifier[3] = nn.Linear(self.mobilenet.classifier[3].in_features, 2)

    def forward(self, x):
        return self.mobilenet(x)


class ShuffleNetBinaryClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ShuffleNetBinaryClassifier, self).__init__()
        self.model = models.shufflenet_v2_x1_0(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 转换为 3 通道
    transforms.RandomRotation(degrees=15),  # 随机旋转 -15 到 15 度
    transforms.RandomResizedCrop(size=(240, 240), scale=(0.8, 1.0)),  # 随机缩放并裁剪
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
    transforms.ToTensor()
])

Vit_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),  # 转换为 3 通道
    transforms.RandomRotation(degrees=15),  # 随机旋转 -15 到 15 度
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # 随机缩放并裁剪
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
    transforms.ToTensor()
])

Inception_transform = transforms.Compose([
    transforms.Resize(299),
    transforms.Grayscale(num_output_channels=3),  # 转换为 3 通道
    transforms.RandomRotation(degrees=15),  # 随机旋转 -15 到 15 度
    transforms.RandomResizedCrop(size=(299, 299), scale=(0.8, 1.0)),  # 随机缩放并裁剪
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 转换为 3 通道
    transforms.ToTensor()
])

Inception_transform_test = transforms.Compose([
    transforms.Resize(299),
    transforms.Grayscale(num_output_channels=3),  # 转换为 3 通道
    transforms.ToTensor()
])

Vit_transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),  # 转换为 3 通道
    transforms.ToTensor()
])


def get_log(name):
    log_manager = Wt_logger_manager(name)
    logger = log_manager.get_logger()
    return logger


def data_init(label, model_name="Densenet"):
    model_label = "L"
    if label == 1:
        label = "label1"
    elif label == 2:
        label = "label2"
        model_label = "C"
    elif label == 3:
        label = "label3"
        model_label = "HIGH"
    data_dir = "data"
    arr_loaded_train = np.load(rf'{data_dir}/{label}/train.npy', allow_pickle=True)
    arr_loaded_train_label = np.load(rf'{data_dir}/{label}/train_label.npy', allow_pickle=True)
    arr_loaded_train_name = np.load(rf'{data_dir}/{label}/train_name.npy', allow_pickle=True)
    arr_loaded_val = np.load(rf'{data_dir}/{label}/val.npy', allow_pickle=True)
    arr_loaded_val_label = np.load(rf'{data_dir}/{label}/val_label.npy', allow_pickle=True)
    arr_loaded_val_name = np.load(rf'{data_dir}/{label}/val_name.npy', allow_pickle=True)
    arr_loaded_test = np.load(rf'{data_dir}/{label}/test.npy', allow_pickle=True)
    arr_loaded_test_label = np.load(rf'{data_dir}/{label}/test_label.npy', allow_pickle=True)
    arr_loaded_test_name = np.load(rf'{data_dir}/{label}/test_name.npy', allow_pickle=True)

    if model_name == "Inception":
        train_dataset = CTDataset(ct_images=arr_loaded_train, labels=arr_loaded_train_label,
                                  names=arr_loaded_train_name,
                                  model_label=model_label,
                                  transform=Inception_transform)
        val_dataset = CTDataset(ct_images=arr_loaded_val, labels=arr_loaded_val_label, names=arr_loaded_val_name,
                                model_label=model_label, transform=Inception_transform_test)
        test_dataset = CTDataset(ct_images=arr_loaded_test, labels=arr_loaded_test_label, names=arr_loaded_test_name,
                                 model_label=model_label, transform=Inception_transform_test)
    elif model_name == "Vit":
        train_dataset = CTDataset(ct_images=arr_loaded_train, labels=arr_loaded_train_label,
                                  names=arr_loaded_train_name,
                                  model_label=model_label,
                                  transform=Vit_transform)
        val_dataset = CTDataset(ct_images=arr_loaded_val, labels=arr_loaded_val_label, names=arr_loaded_val_name,
                                model_label=model_label, transform=Vit_transform_test)
        test_dataset = CTDataset(ct_images=arr_loaded_test, labels=arr_loaded_test_label, names=arr_loaded_test_name,
                                 model_label=model_label, transform=Vit_transform_test)
    else:
        train_dataset = CTDataset(ct_images=arr_loaded_train, labels=arr_loaded_train_label,
                                  names=arr_loaded_train_name,
                                  model_label=model_label,
                                  transform=transform)
        val_dataset = CTDataset(ct_images=arr_loaded_val, labels=arr_loaded_val_label, names=arr_loaded_val_name,
                                model_label=model_label, transform=transform_test)
        test_dataset = CTDataset(ct_images=arr_loaded_test, labels=arr_loaded_test_label, names=arr_loaded_test_name,
                                 model_label=model_label, transform=transform_test)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return train_loader, val_loader, test_loader, label


def train(model, train_loader, criterion, optimizer, device, wt_logger):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    for inputs, labels, names in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        # 记录真实标签和预测分数
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(outputs.softmax(dim=1)[:, 1].cpu().detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    # 计算AUC
    if len(all_labels) != 0:
        epoch_auc = roc_auc_score(all_labels, all_preds)

    return epoch_loss, epoch_acc, epoch_auc, all_labels, all_preds


def test(model, test_loader, criterion, device, wt_logger):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels, names in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            # 记录真实标签和预测分数
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.softmax(dim=1)[:, 1].cpu().detach().numpy())
            wt_logger.info(
                f"\n Test predicted & labels: \n {predicted} \n {labels} \n names:{names}\n loss:{loss} \n correct:{correct}  accuracy: {100 * correct / total}")
    epoch_loss = 0
    epoch_acc = 0
    if total != 0:
        epoch_loss = running_loss / total
        epoch_acc = correct / total
    epoch_auc = 0
    # 计算AUC
    if len(all_labels) != 0:
        epoch_auc = roc_auc_score(all_labels, all_preds)

    return epoch_loss, epoch_acc, epoch_auc, all_labels, all_preds


def train_epoch(model, model_name, train_loader, val_loader, test_loader, label):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    weight = torch.FloatTensor([1, 1])
    if label == "label2":
        weight = torch.FloatTensor([4, 19])
    elif label == "label3":
        weight = torch.FloatTensor([3, 5])
    print(f"label:{label}")

    criterion = nn.CrossEntropyLoss(weight=weight).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, eps=1e-08)
    num_epochs = 100

    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []
    train_auc_list = []
    test_auc_list = []
    train_labels_list = []
    test_labels_list = []
    train_preds_list = []
    test_preds_list = []

    early_stopping = wt_EarlyStopping(patience=10, verbose=True, label=label, model_name=model_name)

    wt_logger = get_log(model_name)
    t_auc = 0
    for epoch in range(num_epochs):
        train_loss, train_acc, train_auc, train_labels, train_preds = train(model, train_loader, criterion, optimizer,
                                                                            device, wt_logger)
        test_loss, test_acc, test_auc, test_labels, test_preds = test(model, test_loader, criterion, device, wt_logger)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        train_auc_list.append(train_auc)
        train_labels_list.append(train_labels)
        train_preds_list.append(train_preds)

        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        test_auc_list.append(test_auc)
        test_labels_list.append(test_labels)
        test_preds_list.append(test_preds)

        wt_logger.info(f"model_name:{model_name}  {label}  Epoch {epoch + 1}/{num_epochs}:")
        wt_logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f}")
        wt_logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}")

        print(f"model_name:{model_name}  Epoch {epoch + 1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}")

        if epoch > 20:
            early_stopping(test_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

    save_pic(model_name, train_loss_list, train_acc_list, train_auc_list, train_labels_list, train_preds_list, label,
             model_label="Train")
    save_pic(model_name, test_loss_list, test_acc_list, test_auc_list, test_labels_list, test_preds_list, label,
             model_label="Test")

    torch.save(model.state_dict(), f"model/{label}/{model_name}.pth")


def save_pic(model_name, loss, acc, auc, labels, preds, label, model_label):
    data_save_path = f"./img/{model_name}/{label}/{model_label}"

    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)
        print(f"Folder '{data_save_path}' created successfully.")

    df_loss = pd.DataFrame(loss)
    df_loss.to_csv(f'{data_save_path}/Loss.csv', index=False)

    df_acc = pd.DataFrame(acc)
    df_acc.to_csv(f'{data_save_path}/Acc.csv', index=False)

    df_auc = pd.DataFrame(auc)
    df_auc.to_csv(f'{data_save_path}/Auc.csv', index=False)
    #
    df_labels = pd.DataFrame(labels)
    df_labels.to_csv(f'{data_save_path}/Labels.csv', index=False)

    df_preds = pd.DataFrame(preds)
    df_preds.to_csv(f'{data_save_path}/Preds.csv', index=False)
    #
    # Loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss, label=f'{model_label} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_label} Loss Curve')
    plt.legend()
    plt.savefig(f'{data_save_path}/Loss.png', dpi=300, bbox_inches='tight')

    # ROC
    fpr, tpr, thresholds = roc_curve(labels[-1], preds[-1])
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc[-1]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_label} Roc')
    plt.legend(loc="lower right")
    plt.savefig(f'{data_save_path}/Roc.png', dpi=300, bbox_inches='tight')

    # Acc
    plt.figure(figsize=(10, 5))
    plt.plot(acc, label=f'{model_label} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.savefig(f'{data_save_path}/Acc.png', dpi=300, bbox_inches='tight')

    # 绘制最后一轮的混淆矩阵
    # 计算并保存混淆矩阵
    cm = confusion_matrix(labels[-1], np.round(preds[-1]))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix {cm[1][0]} {cm[1][1]}')
    plt.savefig(f'{data_save_path}/confusion_matrices.png', dpi=300, bbox_inches='tight')

    # 绘制最后一轮的 Precision-Recall 曲线
    precision, recall, _ = precision_recall_curve(labels[-1], preds[-1])
    average_precision = average_precision_score(labels[-1], preds[-1])
    plt.figure(figsize=(10, 5))
    plt.plot(recall, precision, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(f'{data_save_path}/precision_recall_curves.png', dpi=300, bbox_inches='tight')

    plt.close('all')


def contrast_2d():
    num_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ResNet_18 = ResNet18Classifier(num_classes=num_classes).to(device)
    model_ResNet_50 = ResNet50Classifier(num_classes=num_classes).to(device)
    model_Densenet = DensenetClassifier(num_classes=num_classes).to(device)
    model_Mobilenet = MobilenetClassifier(num_classes=num_classes).to(device)
    model_Vgg = VggClassifier(num_classes=num_classes).to(device)

    train_loader, val_loader, test_loader, label = data_init(1, "Densenet")
    train_epoch(model_ResNet_18, "ResNet_18", train_loader, val_loader, test_loader, label)
    train_epoch(model_ResNet_50, "ResNet_50", train_loader, val_loader, test_loader, label)
    train_epoch(model_Densenet, "Densenet", train_loader, val_loader, test_loader, label)
    train_epoch(model_Mobilenet, "Mobilenet", train_loader, val_loader, test_loader, label)
    train_epoch(model_Vgg, "Vgg19", train_loader, val_loader, test_loader, label)

    train_loader, val_loader, test_loader, label = data_init(2, "Densenet")
    train_epoch(model_ResNet_18, "ResNet_18", train_loader, val_loader, test_loader, label)
    train_epoch(model_ResNet_50, "ResNet_50", train_loader, val_loader, test_loader, label)
    train_epoch(model_Vgg, "Vgg19", train_loader, val_loader, test_loader, label)
    train_epoch(model_Densenet, "Densenet", train_loader, val_loader, test_loader, label)
    train_epoch(model_Mobilenet, "Mobilenet", train_loader, val_loader, test_loader, label)

    train_loader, val_loader, test_loader, label = data_init(3, "Densenet")
    train_epoch(model_ResNet_18, "ResNet_18", train_loader, val_loader, test_loader, label)
    train_epoch(model_ResNet_50, "ResNet_50", train_loader, val_loader, test_loader, label)
    train_epoch(model_Vgg, "Vgg19", train_loader, val_loader, test_loader, label)
    train_epoch(model_Densenet, "Densenet", train_loader, val_loader, test_loader, label)
    train_epoch(model_Mobilenet, "Mobilenet", train_loader, val_loader, test_loader, label)


def contrast_2d_buchong():
    num_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_ResNet_152 = ResNet152Classifier(num_classes=num_classes).to(device)
    model_Shuffle = ShuffleNetBinaryClassifier(num_classes=num_classes).to(device)

    train_loader, val_loader, test_loader, label = data_init(1)
    train_epoch(model_ResNet_152, "ResNet_152", train_loader, val_loader, test_loader, label)
    train_epoch(model_Shuffle, "Shuffle", train_loader, val_loader, test_loader, label)

    train_loader, val_loader, test_loader, label = data_init(2)
    train_epoch(model_ResNet_152, "ResNet_152", train_loader, val_loader, test_loader, label)
    train_epoch(model_Shuffle, "Shuffle", train_loader, val_loader, test_loader, label)

    train_loader, val_loader, test_loader, label = data_init(3)
    train_epoch(model_ResNet_152, "ResNet_152", train_loader, val_loader, test_loader, label)
    train_epoch(model_Shuffle, "Shuffle", train_loader, val_loader, test_loader, label)


num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def contrast_2d_20240903():
    """
    Vgg19:label1、label2、label3
    Densenet:label2、label3
    ResNet101:label1
    """
    model_Vgg = VggClassifier(num_classes=num_classes).to(device)
    model_Densenet = DensenetClassifier(num_classes=num_classes).to(device)
    model_ResNet101 = ResNet101Classifier(num_classes=num_classes).to(device)

    # train_loader, val_loader, test_loader, label = data_init(1)
    # train_epoch(model_Vgg, "Vgg19", train_loader, val_loader, test_loader, label)
    # train_epoch(model_ResNet101, "ResNet101", train_loader, val_loader, test_loader, label)


    # train_loader, val_loader, test_loader, label = data_init(2)
    # train_epoch(model_Vgg, "Vgg19", train_loader, val_loader, test_loader, label)
    # train_epoch(model_Densenet, "Densenet", train_loader, val_loader, test_loader, label)


    train_loader, val_loader, test_loader, label = data_init(3)
    train_epoch(model_Vgg, "Vgg19", train_loader, val_loader, test_loader, label)
    # train_epoch(model_Densenet, "Densenet", train_loader, val_loader, test_loader, label)



if __name__ == '__main__':
    contrast_2d_20240903()
