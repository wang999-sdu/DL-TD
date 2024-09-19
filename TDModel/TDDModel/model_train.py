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

from TDModel.wtCommons.wt_earlys import wt_EarlyStopping
from TDModel.wtliNewModel.data_process import CTDataset
import torchio as tio
from TDModel.wtCommons.wt_log import wt_log
from scipy.signal import savgol_filter
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from PIL import Image

from TDModel.wtliNewModel.wt_contrast import ResNet50Classifier, VitClassifier, ResNet18Classifier, \
    ResNet152Classifier, MobilenetClassifier, DensenetClassifier


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT)
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

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 转换为 3 通道
    transforms.RandomRotation(degrees=15),  # 随机旋转 -15 到 15 度
    transforms.RandomResizedCrop(size=(240, 240), scale=(0.8, 1.0)),  # 随机缩放并裁剪
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 转换为 3 通道
    # transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
    transforms.ToTensor()
])


def data_init(label):
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

    # 创建数据集
    train_dataset = CTDataset(ct_images=arr_loaded_train, labels=arr_loaded_train_label, names=arr_loaded_train_name,
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


def train(model, train_loader, criterion, optimizer, device):
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
        # wt_log.info(
        #     f"Train  predicted & labels: \n {predicted} \n {labels} \n names:{names}\n loss:{loss} \n correct:{correct}  accuracy: {100 * correct / total}")
        # print(
        #     f"Train outputs: {outputs} \n labels: {labels} \n names:{names}\n loss:{loss} \n correct:{correct}  accuracy: {100 * correct / total}")

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    # 计算AUC
    if len(all_labels) != 0:
        epoch_auc = roc_auc_score(all_labels, all_preds)

    return epoch_loss, epoch_acc, epoch_auc, all_labels, all_preds


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels, names in val_loader:
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
            wt_log.info(
                f"\n Val predicted & labels:\n {predicted} \n {labels} \n names:{names}\n loss:{loss} \n correct:{correct}  accuracy: {100 * correct / total}")
            # print(
            #     f"\n Val outputs: {outputs} \n predicted:{predicted} \n labels: {labels} \n names:{names}\n loss:{loss} \n correct:{correct}  accuracy: {100 * correct / total}")

    epoch_loss = 0
    epoch_acc = 0
    if total != 0:
        epoch_loss = running_loss / total
        epoch_acc = correct / total

    # 计算AUC
    epoch_auc = 0
    # 计算AUC
    if len(all_labels) != 0:
        epoch_auc = roc_auc_score(all_labels, all_preds)

    return epoch_loss, epoch_acc, epoch_auc, all_labels, all_preds


def test(model, test_loader, criterion, device):
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
            wt_log.info(
                f"\n Test predicted & labels: \n {predicted} \n {labels} \n names:{names}\n loss:{loss} \n correct:{correct}  accuracy: {100 * correct / total}")
            # print(
            #     f"\n Test outputs: {outputs} \n predicted:{predicted} \n labels: {labels} \n names:{names}\n loss:{loss} \n correct:{correct}  accuracy: {100 * correct / total}")
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


def train_epoch(train_loader, val_loader, test_loader, label):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetClassifier(num_classes=2).to(device)
    weight = torch.FloatTensor([1, 1])
    if label == "label2":
        weight = torch.FloatTensor([1, 4])
    elif label == "label3":
        weight = torch.FloatTensor([1, 2])
    print(f"label:{label}")
    criterion = nn.CrossEntropyLoss(weight=weight).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    num_epochs = 100

    train_loss_list = []
    val_loss_list = []
    test_loss_list = []

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    train_auc_list = []
    val_auc_list = []
    test_auc_list = []

    train_labels_list = []
    val_labels_list = []
    test_labels_list = []

    train_preds_list = []
    val_preds_list = []
    test_preds_list = []

    # 使用示例
    early_stopping = wt_EarlyStopping(patience=16, verbose=True, label=label)

    max_test_auc = 0
    for epoch in range(num_epochs):
        train_loss, train_acc, train_auc, train_labels, train_preds = train(model, train_loader, criterion, optimizer,
                                                                            device)
        val_loss, val_acc, val_auc, val_labels, val_preds = validate(model, val_loader, criterion, device)
        test_loss, test_acc, test_auc, test_labels, test_preds = test(model, test_loader, criterion, device)
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

        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        val_auc_list.append(val_auc)
        val_labels_list.append(val_labels)
        val_preds_list.append(val_preds)

        wt_log.info(f"{label} Epoch {epoch + 1}/{num_epochs}:")
        wt_log.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f}")
        wt_log.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
        wt_log.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}")

        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}")

        if epoch > 20:
            early_stopping(val_loss, model)

            if test_auc > max_test_auc:
                print(f"epoch:{epoch} max_test_auc: {test_auc}")
                max_test_auc = test_auc
                torch.save(model.state_dict(), f"model/{label}/resnet_ct_classifier_with_auc_0830_max.pth")

            if early_stopping.early_stop:
                print("Early stopping")
                break

    save_pic(train_loss_list, train_acc_list, train_auc_list, train_labels_list, train_preds_list, label,
             model_label="Train")
    save_pic(val_loss_list, val_acc_list, val_auc_list, val_labels_list, val_preds_list, label, model_label="Val")
    save_pic(test_loss_list, test_acc_list, test_auc_list, test_labels_list, test_preds_list, label, model_label="Test")

    torch.save(model.state_dict(), f"model/{label}/resnet_ct_classifier_with_auc-0830.pth")


def test_model(test_loader, model_path, num_classes, label, model_label):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 1])).to(device)

    test_loss, test_acc, test_auc, test_labels, test_preds = test(model, test_loader, criterion, device)

    wt_log.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}")
    save_pic([test_loss], [test_acc], [test_auc], [test_labels], [test_preds], label, model_label=model_label)


def save_pic(loss, acc, auc, labels, preds, label, model_label):
    df_loss = pd.DataFrame(loss)
    df_loss.to_csv(f'./img/{label}/{model_label}/Loss.csv', index=False)

    df_acc = pd.DataFrame(acc)
    df_acc.to_csv(f'./img/{label}/{model_label}/Acc.csv', index=False)

    df_auc = pd.DataFrame(auc)
    df_auc.to_csv(f'./img/{label}/{model_label}/Auc.csv', index=False)
    #
    df_labels = pd.DataFrame(labels)
    df_labels.to_csv(f'./img/{label}/{model_label}/Labels.csv', index=False)

    df_preds = pd.DataFrame(preds)
    df_preds.to_csv(f'./img/{label}/{model_label}/Preds.csv', index=False)
    #
    # Loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss, label=f'{model_label} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_label} Loss Curve')
    plt.legend()
    plt.savefig(f'./img/{label}/{model_label}/Loss.png', dpi=300, bbox_inches='tight')

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
    plt.savefig(f'./img/{label}/{model_label}/Roc.png', dpi=300, bbox_inches='tight')

    # Acc
    plt.figure(figsize=(10, 5))
    plt.plot(acc, label=f'{model_label} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.savefig(f'./img/{label}/{model_label}/Acc.png', dpi=300, bbox_inches='tight')

    # 绘制最后一轮的混淆矩阵
    # 计算并保存混淆矩阵
    cm = confusion_matrix(labels[-1], np.round(preds[-1]))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix {cm[1][0]} {cm[1][1]}')
    plt.savefig(f'./img/{label}/{model_label}/confusion_matrices.png', dpi=300, bbox_inches='tight')

    # 绘制最后一轮的 Precision-Recall 曲线
    precision, recall, _ = precision_recall_curve(labels[-1], preds[-1])
    average_precision = average_precision_score(labels[-1], preds[-1])
    plt.figure(figsize=(10, 5))
    plt.plot(recall, precision, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(f'./img/{label}/{model_label}/precision_recall_curves.png', dpi=300, bbox_inches='tight')


def build_dataloader_from_csv(csv_file, label_name):
    import os
    from PIL import Image

    num = 0
    ct_images_train = []
    ct_label_train = []
    ct_train_name = []
    img_path = r"G:\data-CT\csv0828\image_all"
    train_file_list = os.listdir(img_path)
    for index, row in csv_file.iterrows():
        num += 1
        filename = str(row['患者编号'])
        label = str(row[label_name])
        print(f"Processing Train {filename} {num}:{len(csv_file)}  label:{label}")
        a_files = [f for f in train_file_list if f.startswith(filename + "-") and f.endswith('.jpg')]
        for file_name in a_files:
            file_path = os.path.join(img_path, file_name)
            image = Image.open(file_path)
            ct_images_train.append(image)
            ct_label_train.append(label)
            ct_train_name.append(file_name)
    data_array = np.array(ct_images_train)
    label_array = np.array(ct_label_train)
    name_array = np.array(ct_train_name)
    return data_array, label_array, name_array


def cross_validation():
    from sklearn.metrics import roc_auc_score
    from itertools import combinations
    import pandas as pd

    train_path = r"G:\data-CT\csv0828\origin0828.csv"
    img_path = r"G:\data-CT\csv0828\image_all"
    train_list = pd.read_csv(train_path)
    best_roc = 0
    best_institutions = None
    train_list = train_list[train_list["LABEL3"].notna()]
    institutions = train_list['患者来源'].unique()  # 获取所有机构
    # dept_categories = train_list['dept'].unique()  # 获取所有类别

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for combo in combinations(institutions, 2):

        model = ResNetClassifier(num_classes=1).to(device)
        print(f"当前机构组合: {combo}")

        test_set = train_list[train_list['患者来源'].isin(combo)]
        train_set = train_list[~train_list['患者来源'].isin(combo)]
        test_data_array, test_label_array, test_name_array = build_dataloader_from_csv(test_set, "LABEL3")
        train_data_array, train_label_array, train_name_array = build_dataloader_from_csv(train_set, "LABEL3")
        model_label = "HIGH"

        train_dataset = CTDataset(ct_images=train_data_array, labels=train_label_array,
                                  names=train_name_array,
                                  model_label=model_label,
                                  transform=transform)

        test_dataset = CTDataset(ct_images=test_data_array, labels=test_label_array, names=test_name_array,
                                 model_label=model_label,
                                 transform=transform)

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 2])).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        for epoch in range(30):
            train_loss, train_acc, train_auc, train_labels, train_preds = train(model, train_loader, criterion,
                                                                                optimizer,
                                                                                device)
            test_loss, test_acc, test_auc, test_labels, test_preds = test(model, test_loader, criterion, device)
            print(f"epoch:{epoch} \n test_auc:{test_auc}, test_acc:{test_acc}")

            if test_auc > best_roc:
                best_roc = test_auc
                best_institutions = combo
                print(f"Update combo:{best_roc} - {best_institutions}")

    print(f"最佳机构组合: {best_institutions}, ROC AUC: {best_roc:.2f}")


def train_3_label():
    train_loader, val_loader, test_loader, label = data_init(1)
    train_epoch(train_loader, val_loader, test_loader, label)
    train_loader, val_loader, test_loader, label = data_init(2)
    train_epoch(train_loader, val_loader, test_loader, label)
    train_loader, val_loader, test_loader, label = data_init(3)
    train_epoch(train_loader, val_loader, test_loader, label)


def test_3_label():
    _, val_loader, test_loader, _ = data_init(1)
    test_model(val_loader, "model/label1/resnet_ct_classifier_with_auc_0829_max.pth", 2, "label1", "Val")
    test_model(test_loader, "model/label1/resnet_ct_classifier_with_auc_0829_max.pth", 2, "label1", "Test")
    _, val_loader, test_loader, _ = data_init(2)
    test_model(val_loader, "model/label2/resnet_ct_classifier_with_auc_0829_max.pth", 2, "label2", "Val")
    test_model(test_loader, "model/label2/resnet_ct_classifier_with_auc_0829_max.pth", 2, "label2", "Test")
    _, val_loader, test_loader, _ = data_init(3)
    test_model(val_loader, "model/label3/resnet_ct_classifier_with_auc_0829_max.pth", 2, "label3", "Val")
    test_model(test_loader, "model/label3/resnet_ct_classifier_with_auc_0829_max.pth", 2, "label3", "Test")


def test_with_humans():
    import os
    """
    最终于人类进行对比试验
    """
    img_path = r"G:\data-CT\csv0828\exp69_output"
    img_list = os.listdir(img_path)
    # label1_model_path = r"model\label1-ResNet_50-checkpoint.pt"
    # label2_model_path = r"model\label2-ResNet_50-checkpoint.pt"
    # label3_model_path = r"model\label3-ResNet_50-checkpoint.pt"
    label1_model_path = r"model-resnet\label1\checkpoint.pt"
    label2_model_path = r"model-resnet\label2\resnet_ct_classifier_with_auc-0830.pth"
    label3_model_path = r"model-resnet\label3\resnet_ct_classifier_with_auc-0830.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label1_model = ResNetClassifier(num_classes=2).to(device)
    label1_model.load_state_dict(torch.load(label1_model_path))
    label2_model = ResNetClassifier(num_classes=2).to(device)
    label2_model.load_state_dict(torch.load(label2_model_path))
    label3_model = ResNetClassifier(num_classes=2).to(device)
    label3_model.load_state_dict(torch.load(label3_model_path))
    label1_model.eval()
    label2_model.eval()
    label3_model.eval()
    name_list = []
    label1_list = []
    label2_list = []
    label3_list = []

    for img_name in img_list:
        file_path = os.path.join(img_path, img_name)
        image = Image.open(file_path)
        image = transform_test(image).to(device)
        image = image.unsqueeze(0)

        outputs_label1 = label1_model(image)
        _, predicted_label1 = torch.max(outputs_label1, 1)

        outputs_label2 = label2_model(image)
        _, predicted_label2 = torch.max(outputs_label2, 1)

        outputs_label3 = label3_model(image)
        _, predicted_label3 = torch.max(outputs_label3, 1)
        img_name = img_name.split('.')[0]
        name_list.append(img_name)
        label1_list.append(predicted_label1.item())
        label2_list.append(predicted_label2.item())
        label3_list.append(predicted_label3.item())
        print(img_name, predicted_label1.item(), predicted_label2.item(), predicted_label3.item())
    data = {
        'name': name_list,
        'label1': label1_list,
        'label2': label2_list,
        'label3': label3_list
    }
    df = pd.DataFrame(data)

    # 将 DataFrame 对象保存为 CSV 文件
    df.to_csv('output.csv', index=False)

if __name__ == '__main__':
    # train_3_label()
    # train_loader, val_loader, test_loader, label = data_init(1)
    # train_epoch(train_loader, val_loader, test_loader, label)
    # train_loader, val_loader, test_loader, label = data_init(3)
    # train_epoch(train_loader, val_loader, test_loader, label)
    # train_loader, val_loader, test_loader, _ = data_init(3)
    # test_model(test_loader, "model/label3/resnet_ct_classifier_with_auc_0830_max.pth", 2, "label3", "Test")
    # test_model(val_loader, "model/label3/resnet_ct_classifier_with_auc_0830_max.pth", 2, "label3", "Val")
    test_with_humans()