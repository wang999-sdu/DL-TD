import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import os
import pandas as pd
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split


def load_ct_images_1(labell=1):
    """
    input: ct Data
    process: 裁剪（异形label）+标准化+调整大小
    ouput: array tensor.pth
    """
    image_path = r"G:\data-CT\csv0828\image_train_val_0829"
    test_image_path = r"G:\data-CT\csv0828\image_test_0829"
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
    image_list = os.listdir(image_path)
    test_image_list = os.listdir(test_image_path)

    # df_train = df_all[(df_all["test"] == 0) & (df_all[label_name].notna())]

    df_train = pd.read_csv("data/train_set.csv")
    df_val = pd.read_csv("data/val_set.csv")
    df_test = pd.read_csv("data/test_set.csv")

    df_train = df_train[df_train[label_name].notna()]
    df_val = df_val[df_val[label_name].notna()]
    df_test = df_test[df_test[label_name].notna()]

    # =======  Train  ========
    # for index, row in df_train.iterrows():
    #     num += 1
    #     filename = str(row['患者编号'])
    #     label = str(row[label_name])
    #     print(f"Processing Train {filename} {num}:{len(df_train)}  label:{label}")
    #     a_files = [f for f in image_list if f.startswith(filename + "-") and f.endswith('.jpg')]
    #     for file_name in a_files:
    #         file_path = os.path.join(image_path, file_name)
    #         image = Image.open(file_path)
    #         ct_images_train.append(image)
    #         ct_label_train.append(label)
    #         ct_train_name.append(file_name)
    # ct_data_array = np.array(ct_images_train)
    # ct_label_array = np.array(ct_label_train)
    # ct_train_name = np.array(ct_train_name)
    # np.save(rf'data/{label_name}/train.npy', ct_data_array)
    # np.save(rf'data/{label_name}/train_label.npy', ct_label_array)
    # np.save(rf'data/{label_name}/train_name.npy', ct_train_name)

    # =======  Val  ========
    num = 0
    for index, row in df_val.iterrows():
        num += 1
        filename = str(row['患者编号'])
        label = str(row[label_name])
        print(f"Processing Val {filename} {num}:{len(df_val)}  label:{label}")
        a_files = [f for f in image_list if f.startswith(filename + "-") and f.endswith('.jpg')]
        for file_name in a_files:
            file_path = os.path.join(r"G:\data-CT\20240826xxl\image_tumors_origin", file_name)
            image = Image.open(file_path)
            ct_images_val.append(image)
            ct_label_val.append(label)
            ct_val_name.append(file_name)
    ct_data_array = np.array(ct_images_val)
    ct_label_array = np.array(ct_label_val)
    ct_val_name = np.array(ct_val_name)
    np.save(rf'data/{label_name}/val.npy', ct_data_array)
    np.save(rf'data/{label_name}/val_label.npy', ct_label_array)
    np.save(rf'data/{label_name}/val_name.npy', ct_val_name)

    # =======  Test  ========
    # num = 0
    # for index, row in df_test.iterrows():
    #     num += 1
    #     filename = str(row['患者编号']).lower() + ".jpg"
    #     label = str(row[label_name])
    #     print(f"Processing Test {filename} {num}:{len(df_test)}  label:{label}")
    #     file_path = os.path.join(test_image_path, filename)
    #     image = Image.open(file_path)
    #     ct_images_test.append(image)
    #     ct_label_test.append(label)
    #     ct_name_test.append(filename)
    #     # a_files = [f for f in test_image_list if f.startswith(filename) and f.endswith('.jpg')]
    #     # for file_name in a_files:
    #     #     file_path = os.path.join(test_image_path, file_name)
    #     #     image = Image.open(file_path)
    #     #     ct_images_test.append(image)
    #     #     ct_label_test.append(label)
    #     #     ct_name_test.append(file_name)
    # ct_data_array = np.array(ct_images_test)
    # ct_label_array = np.array(ct_label_test)
    # ct_name_array = np.array(ct_name_test)
    # np.save(rf'data/{label_name}/test.npy', ct_data_array)
    # np.save(rf'data/{label_name}/test_label.npy', ct_label_array)
    # np.save(rf'data/{label_name}/test_name.npy', ct_name_array)


# 定义一个函数用于分割数据集
def split_data(df):
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    return train, test


def divide_dataset():
    """
    划分训练集、验证集
    """
    df = pd.read_csv(r'G:\data-CT\csv0828\origin0828.csv')
    df_test = df[df['test'] == 1]
    df = df[df['test'] == 0]
    # 分别提取出 Label1 为 N 和 L 的数据
    df_label1_N = df[df['label1'] == 'N']
    df_label1_L = df[df['label1'] == 'L']

    # 从 Label1 为 L 中提取出 Label2 为 C 和 L 的数据
    df_label2_Other = df_label1_L[df_label1_L['label2'] != 'L']
    df_label2_L = df_label1_L[df_label1_L['label2'] == 'L']

    # 从 Label2 为 L 中提取出 Label3 为 HIGH 和 LOW 的数据
    df_label3_HIGH = df_label2_L[df_label2_L['label3'] == 'HIGH']
    df_label3_LOW = df_label2_L[df_label2_L['label3'] == 'LOW']

    # 对每个分组进行分割
    train_label1_N, val_label1_N = split_data(df_label1_N)
    train_label2_Other, val_label2_Other = split_data(df_label2_Other)
    train_label3_HIGH, val_label3_HIGH = split_data(df_label3_HIGH)
    train_label3_LOW, val_label3_LOW = split_data(df_label3_LOW)

    # 合并训练集和测试集
    train_set = pd.concat([train_label1_N, train_label2_Other, train_label3_HIGH, train_label3_LOW])
    val_set = pd.concat([val_label1_N, val_label2_Other, val_label3_HIGH, val_label3_LOW])

    train_set["train"] = 1
    val_set["train"] = 0
    train_set['患者编号'] = train_set['患者编号'].str.lower()
    val_set['患者编号'] = val_set['患者编号'].str.lower()
    df_test['患者编号'] = df_test['患者编号'].str.lower()

    train_set.to_csv('train_set.csv', index=False)
    val_set.to_csv('val_set.csv', index=False)
    df_test.to_csv('test_set.csv', index=False)


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
        if self.transform:
            image = self.transform(image)

        return image, label, name, "thymoma"


if __name__ == '__main__':
    # load_ct_images_1(1)
    load_ct_images_1(3)
    # load_ct_images_1(3)
    # divide_dataset()
