import nibabel as nib
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import pandas as pd
import torchvision.transforms as transform
import SimpleITK as sitk
import skimage
import monai
import pydicom
import torch
from skimage.measure import label, regionprops
from skimage.transform import resize
import concurrent.futures
from tqdm import tqdm  # 进度条库
from sklearn.model_selection import train_test_split


def split_data(df):
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    return train, test


def divide_data():
    """
    划分训练集、验证集
    """
    df = pd.read_csv('path_to_your_file.csv')
    # 分别提取出 Label1 为 N 和 L 的数据
    df_label1_N = df[df['label1'] == 'N']
    df_label1_L = df[df['label1'] == 'L']
    # 从 Label1 为 L 中提取出 Label2 为 C 和 L 的数据
    df_label2_C = df_label1_L[df_label1_L['label2'] == 'C']
    df_label2_L = df_label1_L[df_label1_L['label2'] == 'L']
    # 从 Label2 为 L 中提取出 Label3 为 HIGH 和 LOW 的数据
    df_label3_HIGH = df_label2_L[df_label2_L['label3'] == 'HIGH']
    df_label3_LOW = df_label2_L[df_label2_L['label3'] == 'LOW']

    # 对每个分组进行分割
    train_label1_N, val_label1_N = split_data(df_label1_N)
    train_label2_C, val_label2_C = split_data(df_label2_C)
    train_label3_HIGH, val_label3_HIGH = split_data(df_label3_HIGH)
    train_label3_LOW, val_label3_LOW = split_data(df_label3_LOW)

    # 合并训练集和测试集
    train_set = pd.concat([train_label1_N, train_label2_C, train_label3_HIGH, train_label3_LOW])
    val_set = pd.concat([val_label1_N, val_label2_C, val_label3_HIGH, val_label3_LOW])

    train_set.to_csv('train_set.csv', index=False)
    val_set.to_csv('test_set.csv', index=False)


def convert_file_to_lower():
    """
    文件夹中的文件全部改成小写
    """
    folder_path = r'G:\data-CT\20240826xxl\image_tumors_test'

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # 获取文件名的小写版本
        lower_filename = filename.lower()
        lower_file_path = os.path.join(folder_path, lower_filename)
        os.rename(file_path, lower_file_path)
    print("All filenames have been converted to lowercase.")


def cutct2img_test():
    # 20240808 ct转img，找肿瘤区域，每个ct转出2类，一类是根据肿瘤大小切图，一类是只有肿瘤的截面
    # 3个img，肿瘤最大截面+上下一层，分别是正常分割、分割去除边缘、分割增加边缘
    img_path = r"G:\data-CT\QD\QD-IMAGE"
    label_path = r"G:\data-CT\QD\QD-LABEL"
    test_path = r"data\test_set.csv"
    output_image_path = r"G:\data-CT\csv0828\image_test_0829"

    test_list = pd.read_csv(test_path)
    progress_queue = tqdm(total=len(test_list), desc="Processing")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for index, row in test_list.iterrows():
        name = row['患者编号']
        print(f"Processing {index} {name} :{len(test_list)}")
        nii_path = os.path.join(img_path, name + "_0000.nii.gz")
        mask_path = os.path.join(label_path, name + "_0000.nii.gz")
        if not os.path.exists(nii_path):
            nii_path = os.path.join(r"G:\data-CT\20240826xxl\image", name.lower() + ".nii.gz")
            mask_path = os.path.join(r"G:\data-CT\20240826xxl\label", name.lower() + ".nii.gz")
        img_nib = nib.load(nii_path)
        label_nib = nib.load(mask_path)
        img_ct_data = img_nib.get_fdata()  # Hu data
        if len(img_ct_data.shape) == 5:
            img_ct_data = img_ct_data[:, :, :, 0, 0]
        if len(img_ct_data.shape) == 4:
            img_ct_data = img_ct_data[:, :, :, 0]
        label_ct_data = label_nib.get_fdata()
        non_zeros_idx = np.where(label_ct_data != 0)
        max_tumor_num = Counter(non_zeros_idx[2]).most_common(1)  # 找到分割的面积最大的图片
        [max_x, max_y, max_z] = np.max(np.array(non_zeros_idx), axis=1)
        [min_x, min_y, min_z] = np.min(np.array(non_zeros_idx), axis=1)
        data_img = np.clip(img_ct_data, -160, 240)
        data_img1 = data_img[:, :, max_tumor_num[0][0]]

        # for i in range(max_tumor_num[0][0] - 2, max_tumor_num[0][0] + 3):
        #     if i >= max_z or i <= min_z:
        #         continue
        #     data_img1 = data_img[:, :, i]
        #     for item_x in range(512):
        #         for item_y in range(512):
        #             if label_ct_data[item_x, item_y, i] == 0:
        #                 data_img1[item_x, item_y] = 0
        #     data_img1 = resize(data_img1, (240, 240), preserve_range=True)
        #     resized_slice_clahe = clahe.apply(np.array(data_img1, dtype=np.uint8))
        #     resized_slice_clahe = Image.fromarray(resized_slice_clahe)
        #     resized_slice_clahe = resized_slice_clahe.convert("L")
        #     resized_slice_clahe = resized_slice_clahe.rotate(-90)
        #     resized_slice_clahe.save(output_image_path + rf"/{name.lower()}-{i}.jpg")

        for item_x in range(512):
            for item_y in range(512):
                if label_ct_data[item_x, item_y, max_tumor_num[0][0]] == 0:
                    data_img1[item_x, item_y] = 0
        data_img1 = resize(data_img1, (240, 240), preserve_range=True)
        resized_slice_clahe = clahe.apply(np.array(data_img1, dtype=np.uint8))
        resized_slice_clahe = Image.fromarray(resized_slice_clahe)
        resized_slice_clahe = resized_slice_clahe.convert("L")
        resized_slice_clahe = resized_slice_clahe.rotate(-90)
        resized_slice_clahe.save(output_image_path + rf"/{name.lower()}.jpg")
        progress_queue.update(1)


def cutct2img_20240807():
    # 20240808 ct转img，找肿瘤区域，每个ct转出2类，一类是根据肿瘤大小切图，一类是只有肿瘤的截面
    # 3个img，肿瘤最大截面+上下一层，分别是正常分割、分割去除边缘、分割增加边缘
    root_path = r"G:\data-CT"
    img_path = root_path + r"\20240826xxl\image"
    label_path = root_path + r"\20240826xxl\label"
    train_path = r"data\train_set.csv"
    val_path = r"data\val_set.csv"
    output_image_path = root_path + r"\csv0828\image_all_0829"

    train_list = pd.read_csv(val_path)
    progress_queue = tqdm(total=len(train_list), desc="Processing")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for index, row in train_list.iterrows():
        name = row['患者编号']
        print(f"Processing {index} {name} :{len(train_list)}")
        nii_path = os.path.join(img_path, name + ".nii.gz")
        mask_path = os.path.join(label_path, name + ".nii.gz")
        img_nib = nib.load(nii_path)
        label_nib = nib.load(mask_path)
        img_ct_data = img_nib.get_fdata()  # Hu data
        if len(img_ct_data.shape) == 5:
            img_ct_data = img_ct_data[:, :, :, 0, 0]
        if len(img_ct_data.shape) == 4:
            img_ct_data = img_ct_data[:, :, :, 0]
        label_ct_data = label_nib.get_fdata()
        non_zeros_idx = np.where(label_ct_data != 0)
        max_tumor_num = Counter(non_zeros_idx[2]).most_common(1)  # 找到分割的面积最大的图片
        [max_x, max_y, max_z] = np.max(np.array(non_zeros_idx), axis=1)
        [min_x, min_y, min_z] = np.min(np.array(non_zeros_idx), axis=1)

        data_img = np.clip(img_ct_data, -160, 240)

        for i in range(max_tumor_num[0][0] - 2, max_tumor_num[0][0] + 3):
            if i >= max_z or i <= min_z:
                continue
            data_img1 = data_img[:, :, i]
            for item_x in range(512):
                for item_y in range(512):
                    if label_ct_data[item_x, item_y, i] == 0:
                        data_img1[item_x, item_y] = 0
            data_img1 = resize(data_img1, (240, 240), preserve_range=True)
            resized_slice_clahe = clahe.apply(np.array(data_img1, dtype=np.uint8))
            resized_slice_clahe = Image.fromarray(resized_slice_clahe)
            resized_slice_clahe = resized_slice_clahe.convert("L")
            resized_slice_clahe = resized_slice_clahe.rotate(-90)
            resized_slice_clahe.save(output_image_path + rf"/{name}-{i}.jpg")
        progress_queue.update(1)



if __name__ == '__main__':
    cutct2img_test()
