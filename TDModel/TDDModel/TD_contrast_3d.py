import nibabel as nb
from sklearn.model_selection import train_test_split
from nibabel.viewers import OrthoSlicer3D
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
from TDModel.TDCommons.wt_earlys import wt_EarlyStopping
from TDModel.TDDModel.data_process import CTDataset
import torchio as tio
from TDModel.TDCommons.wt_log import Wt_logger_manager
from scipy.signal import savgol_filter
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from sklearn.svm import SVC
from TDModel.vit_pytorch import vit_3d
from models import resnet


def rename_files(directory):
    for filename in os.listdir(directory):
        # 将文件名转换为小写
        new_filename = filename.lower()
        # 移除 '_0000' 中部或末尾部分
        new_filename = new_filename.replace('_0000', '')
        # 获取完整路径
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        # 重命名文件
        os.rename(old_file, new_file)


def extract_roi(ct_tensor, mask_tensor):
    non_zeros_idx = np.where(mask_tensor != 0)
    [max_x, max_y, max_z] = np.max(np.array(non_zeros_idx), axis=1)
    [min_x, min_y, min_z] = np.min(np.array(non_zeros_idx), axis=1)
    print(f"current_shape122: {min_x, min_y, min_z},{max_x, max_y, max_z}")
    print(f"mask_tensor: {mask_tensor.shape[2]}")

    data_arr = np.zeros((224, 224, 112))

    # data_arr = ct_tensor[min_x - 10:min_x + 214, min_y - 10:min_y + 214, min_z - 10:min_z + 102]

    x, y, z = data_arr.shape
    for item_x in range(x):
        for item_y in range(y):
            for item_z in range(z):
                min_xx = min_x - 10 + item_x
                min_yy = min_y - 10 + item_y
                min_zz = min_z - 10 + item_z
                if min_xx >= mask_tensor.shape[0]:
                    break
                if min_yy >= mask_tensor.shape[1]:
                    break
                if min_zz >= mask_tensor.shape[2]:
                    break
                if min_zz < 0:
                    min_zz = 0
                if mask_tensor[min_xx, min_yy, min_zz] != 0:
                    data_arr[item_x, item_y, item_z] = ct_tensor[min_xx, min_yy, min_zz]

    return data_arr


def data_preprocessing(img_path, img_label_path):
    """
    分割nii原始文件，根据label中不为0的数据
    """
    # img_path = r"G:\data-CT\20240826xxl\image"
    # img_label_path = r"G:\data-CT\20240826xxl\label"
    nii_seg_path = r"G:\data-CT\csv0828\ct_data"
    num = 0
    img_list = os.listdir(img_path)
    for item in img_list:
        num += 1
        print(f"num: {num}:{len(img_list)}, item:{item}")
        nii_path = os.path.join(img_path, item)
        mask_path = os.path.join(img_label_path, item)
        if not os.path.exists(nii_path) or not os.path.exists(mask_path):
            continue
        img_nib = nb.load(nii_path)
        img_affine = img_nib.affine
        img_ct_header = img_nib.header

        img_ct_header['pixdim'][1:4] = [1.0, 1.0, 1.0]
        # img_ct_header.set_zooms((1.0, 1.0, 1.0))
        img_ct_data = img_nib.get_fdata()  # Hu data
        label_nib = nb.load(mask_path)
        if len(img_ct_data.shape) == 5:
            img_ct_data = img_ct_data[:, :, :, 0, 0]
        if len(img_ct_data.shape) == 4:
            img_ct_data = img_ct_data[:, :, :, 0]
        label_ct_data = label_nib.get_fdata()
        img_ct_data = extract_roi(img_ct_data, label_ct_data)
        new_ct_file = nb.Nifti1Image(img_ct_data, img_affine, img_ct_header)
        nb.save(new_ct_file, nii_seg_path + rf"\{item}")


def data_cut():
    """
    把ct原始数据文件，根据label分割出来
    """
    # img_path = r"G:\data-CT\20240826xxl\image"
    # img_label_path = r"G:\data-CT\20240826xxl\label"
    # data_preprocessing(img_path, img_label_path)

    img_path = r"G:\data-CT\001-xxl\QD\QD-IMAGE"
    img_label_path = r"G:\data-CT\001-xxl\QD\QD-LABEL"
    data_preprocessing(img_path, img_label_path)


def data_save_npy(labell=1):
    """
    读取分割出来的文件，根据train，val，test转成numpy
    """
    ct_path = r"G:\data-CT\csv0828\ct_data"
    ct_images_train = []
    ct_images_val = []
    ct_images_test = []
    ct_label_train = []
    ct_label_val = []
    ct_label_test = []
    ct_name_test = []
    label_images = []
    ct_train_name = []
    ct_val_name = []
    num = 0
    label_name = ""
    if labell == 1:
        label_name = "label1"
    elif labell == 2:
        label_name = "label2"
    elif labell == 3:
        label_name = "label3"
    image_list = os.listdir(ct_path)

    # df_train = df_all[(df_all["test"] == 0) & (df_all[label_name].notna())]

    df_train = pd.read_csv("data/train_set.csv")
    # df_val = pd.read_csv("data/val_set.csv")
    df_test = pd.read_csv("data/test_set.csv")

    df_train = df_train[df_train[label_name].notna()]
    # df_val = df_val[df_val[label_name].notna()]
    df_test = df_test[df_test[label_name].notna()]

    # =======  Train  ========
    for index, row in df_train.iterrows():
        num += 1
        filename = str(row['患者编号'])
        label = str(row[label_name])
        print(f"Processing Train {filename} {num}:{len(df_train)}  label:{label}")
        a_files = [f for f in image_list if f.startswith(filename) and f.endswith('.nii.gz')]
        for file_name in a_files:
            file_path = os.path.join(ct_path, file_name)
            img_nib = nb.load(file_path)
            image = img_nib.get_fdata()
            ct_images_train.append(image)
            ct_label_train.append(label)
            ct_train_name.append(file_name)
    ct_data_array = np.array(ct_images_train)
    ct_label_array = np.array(ct_label_train)
    ct_train_name = np.array(ct_train_name)
    np.save(rf'data/3d_{label_name}/train.npy', ct_data_array)
    np.save(rf'data/3d_{label_name}/train_label.npy', ct_label_array)
    np.save(rf'data/3d_{label_name}/train_name.npy', ct_train_name)

    # =======  Test  ========
    num = 0
    for index, row in df_test.iterrows():
        num += 1
        filename = str(row['患者编号']).lower() + ".nii.gz"
        label = str(row[label_name])
        print(f"Processing Test {filename} {num}:{len(df_test)}  label:{label}")
        file_path = os.path.join(ct_path, filename)
        img_nib = nb.load(file_path)
        image = img_nib.get_fdata()
        ct_images_test.append(image)
        ct_label_test.append(label)
        ct_name_test.append(filename)
    ct_data_array = np.array(ct_images_test)
    ct_label_array = np.array(ct_label_test)
    ct_name_array = np.array(ct_name_test)
    np.save(rf'data/3d_{label_name}/test.npy', ct_data_array)
    np.save(rf'data/3d_{label_name}/test_label.npy', ct_label_array)
    np.save(rf'data/3d_{label_name}/test_name.npy', ct_name_array)


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
    arr_loaded_train = np.load(rf'{data_dir}/3d_{label}/train.npy', allow_pickle=True)
    arr_loaded_train_label = np.load(rf'{data_dir}/3d_{label}/train_label.npy', allow_pickle=True)
    arr_loaded_train_name = np.load(rf'{data_dir}/3d_{label}/train_name.npy', allow_pickle=True)
    arr_loaded_test = np.load(rf'{data_dir}/3d_{label}/test.npy', allow_pickle=True)
    arr_loaded_test_label = np.load(rf'{data_dir}/3d_{label}/test_label.npy', allow_pickle=True)
    arr_loaded_test_name = np.load(rf'{data_dir}/3d_{label}/test_name.npy', allow_pickle=True)

    # 创建数据集
    train_dataset = CTDataset(ct_images=arr_loaded_train, labels=arr_loaded_train_label,
                              names=arr_loaded_train_name,
                              model_label=model_label)
    test_dataset = CTDataset(ct_images=arr_loaded_test, labels=arr_loaded_test_label, names=arr_loaded_test_name,
                             model_label=model_label)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return train_loader, test_loader, label


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


def train_epoch(model, model_name, train_loader, test_loader, label):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    weight = torch.FloatTensor([1, 1])
    if label == "label2":
        weight = torch.FloatTensor([1, 4])
    elif label == "label3":
        weight = torch.FloatTensor([1, 2])
    print(f"label:{label}")

    criterion = nn.CrossEntropyLoss(weight=weight).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
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

    early_stopping = wt_EarlyStopping(patience=5, verbose=True, label=label, model_name=model_name)

    wt_logger = get_log(model_name)

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
            early_stopping(train_loss, model)

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


num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def contrast_vit():
    model_model_Vit_3d = vit_3d.ViT(image_size=224,
                                    frames=112,
                                    image_patch_size=16,
                                    frame_patch_size=4,
                                    num_classes=2,
                                    channels=1,
                                    dim=256,
                                    depth=6,
                                    heads=8,
                                    mlp_dim=2048,
                                    dropout=0.1,
                                    emb_dropout=0.1).to(device)
    train_loader, test_loader, label = data_init(1)
    train_epoch(model_model_Vit_3d, "model_Vit_3d", train_loader, test_loader, label)

    train_loader, test_loader, label = data_init(2)
    train_epoch(model_model_Vit_3d, "model_Vit_3d", train_loader, test_loader, label)

    train_loader, test_loader, label = data_init(3)
    train_epoch(model_model_Vit_3d, "model_Vit_3d", train_loader, test_loader, label)


def contrast_3d():
    model_ResNet101_3d = resnet.resnet101(
        sample_input_W=224,
        sample_input_H=224,
        sample_input_D=112,
        num_seg_classes=2,
    ).to(device)
    model_ResNet101_3d.conv_seg = nn.Sequential(
        nn.AdaptiveAvgPool3d((1, 1, 1)),
        nn.Flatten(),
        nn.Linear(in_features=2048, out_features=2, bias=True),
    )

    train_loader, test_loader, label = data_init(1)
    train_epoch(model_ResNet101_3d, "ResNet101_3d", train_loader, test_loader, label)

    train_loader, test_loader, label = data_init(2)
    train_epoch(model_ResNet101_3d, "ResNet101_3d", train_loader, test_loader, label)

    train_loader, test_loader, label = data_init(3)
    train_epoch(model_ResNet101_3d, "ResNet101_3d", train_loader, test_loader, label)


class CTDataset(Dataset):
    def __init__(self, ct_images, labels, names, model_label="L", transform=None):
        self.model_label = model_label
        self.ct_images = ct_images  # 预处理后的CT图像列表
        self.labels = labels  # 对应的标签
        self.names = names
        self.transform = transform  # 数据增强等转换

    def __len__(self):
        return len(self.ct_images)

    def __getitem__(self, idx):
        image = self.ct_images[idx]
        label = self.labels[idx]
        name = self.names[idx]
        if label == self.model_label:
            label = 1
        else:
            label = 0
        image = torch.tensor(image).unsqueeze(0).float()  # 添加通道维度 [C, D, H, W]
        return image, label, name


if __name__ == '__main__':
    data_save_npy(2)
    data_save_npy(3)
