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


def convert_file_to_lower():
    folder_path = r'G:\data-CT\20240826xxl\image_tumors_test'

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 构造完整的文件路径
        file_path = os.path.join(folder_path, filename)

        # 获取文件名的小写版本
        lower_filename = filename.lower()

        # 构造新的文件路径
        lower_file_path = os.path.join(folder_path, lower_filename)

        # 重命名文件
        os.rename(file_path, lower_file_path)
    print("All filenames have been converted to lowercase.")


def cutct2img_test():
    # 20240808 ct转img，找肿瘤区域，每个ct转出2类，一类是根据肿瘤大小切图，一类是只有肿瘤的截面
    # 3个img，肿瘤最大截面+上下一层，分别是正常分割、分割去除边缘、分割增加边缘
    img_path = r"G:\data-CT\QD\QD-IMAGE"
    label_path = r"G:\data-CT\QD\QD-LABEL"
    test_path = r"G:\data-CT\test.csv"
    output_image_path = r"G:\data-CT\20240826xxl\image_tumors_test2"

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
            resized_slice_clahe.save(output_image_path + rf"/{name.lower()}-{i}.jpg")


        # for item_x in range(512):
        #     for item_y in range(512):
        #         if label_ct_data[item_x, item_y, max_tumor_num[0][0]] == 0:
        #             data_img1[item_x, item_y] = 0
        # data_img1 = resize(data_img1, (240, 240), preserve_range=True)
        # resized_slice_clahe = clahe.apply(np.array(data_img1, dtype=np.uint8))
        # resized_slice_clahe = Image.fromarray(resized_slice_clahe)
        # resized_slice_clahe = resized_slice_clahe.convert("L")
        # resized_slice_clahe = resized_slice_clahe.rotate(-90)
        # resized_slice_clahe.save(output_image_path + rf"/{name.lower()}.jpg")
        progress_queue.update(1)


def cutct2img_20240807():
    # 20240808 ct转img，找肿瘤区域，每个ct转出2类，一类是根据肿瘤大小切图，一类是只有肿瘤的截面
    # 3个img，肿瘤最大截面+上下一层，分别是正常分割、分割去除边缘、分割增加边缘
    root_path = r"G:\data-CT"
    img_path = root_path + r"\20240826xxl\image"
    label_path = root_path + r"\20240826xxl\label"
    train_path = root_path + r"\train.csv"
    test_path = root_path + r"\test.csv"
    output_image_path = root_path + r"\20240826xxl\image_tumors_origin"

    train_list = pd.read_csv(train_path)
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

        # # 1: 找到最大肿瘤截面，包括上下两层
        # data_img = Image.fromarray(img_ct_data[:, :, max_tumor_num[0][0]])
        # data_img1 = data_img.convert("L")
        # data_img1.save(pic_root + rf"/1/{item}-1.jpg")
        # data_img = Image.fromarray(img_ct_data[:, :, max_tumor_num[0][0] - 1])
        # data_img2 = data_img.convert("L")
        # data_img2.save(pic_root + rf"/1/{item}-2.png")
        # data_img = Image.fromarray(img_ct_data[:, :, max_tumor_num[0][0] + 1])
        # data_img3 = data_img.convert("L")
        # data_img3.save(pic_root + rf"/1/{item}-3.png")
        #
        # # 2：找到分割出来最大肿瘤
        # data_tumor_arr1 = np.zeros((224, 224))  # 正常裁剪
        # data_tumor_arr2 = np.zeros((224, 224))  # 只保留大于0的ct值
        # data_tumor_arr3 = np.zeros((224, 224))  # 边缘减少2像素
        # data_tumor_arr4 = np.zeros((224, 224))  # 边缘增加2像素
        # data_tumor_arr5 = np.zeros((224, 224))  # 边缘增加2像素
        # data_tumor_arr6 = np.zeros((224, 224))  # 边缘增加2像素
        # for i in range(224):
        #     for j in range(224):
        #         # 裁剪的不光包含肿瘤
        #         if label_ct_data[min_x - 10 + i, min_y - 10 + j, max_tumor_num[0][0]] != 0:
        #             data_tumor_arr1[i, j] = img_ct_data[min_x - 10 + i, min_y - 10 + j, max_tumor_num[0][0]]
        #         if label_ct_data[min_x - 10 + i, min_y - 10 + j, max_tumor_num[0][0] - 1] != 0:
        #             data_tumor_arr2[i, j] = img_ct_data[min_x - 10 + i, min_y - 10 + j, max_tumor_num[0][0] - 1]
        #         if label_ct_data[min_x - 10 + i, min_y - 10 + j, max_tumor_num[0][0] + 1] != 0:
        #             data_tumor_arr3[i, j] = img_ct_data[min_x - 10 + i, min_y - 10 + j, max_tumor_num[0][0] + 1]
        # data_tumor_arr1 = Image.fromarray(data_tumor_arr1)
        # data_tumor_arr2 = Image.fromarray(data_tumor_arr2)
        # data_tumor_arr3 = Image.fromarray(data_tumor_arr3)
        # data_tumor_arr4 = data_tumor_arr1.rotate(90)
        # data_tumor_arr5 = data_tumor_arr1.rotate(-90)
        # data_tumor_arr6 = data_tumor_arr1.rotate(180)
        # data_tumor_arr1 = data_tumor_arr1.convert("L")
        # data_tumor_arr2 = data_tumor_arr2.convert("L")
        # data_tumor_arr3 = data_tumor_arr3.convert("L")
        # data_tumor_arr4 = data_tumor_arr4.convert("L")
        # data_tumor_arr5 = data_tumor_arr5.convert("L")
        # data_tumor_arr6 = data_tumor_arr6.convert("L")
        # data_tumor_arr1.save(pic_root + rf"/2/{item}-1.png")  # 肿瘤图像最大横截面
        # data_tumor_arr2.save(pic_root + rf"/2/{item}-2.png")  # 肿瘤图像最大横截面，只保留ct值大于0
        # data_tumor_arr3.save(pic_root + rf"/2/{item}-3.png")  # 肿瘤图像最大横截面，-1的截面
        # data_tumor_arr4.save(pic_root + rf"/2/{item}-4.png")
        # data_tumor_arr5.save(pic_root + rf"/2/{item}-5.png")
        # data_tumor_arr6.save(pic_root + rf"/2/{item}-6.png")
        # img1.save(png_save_path + rf"/{item}.png")


if __name__ == '__main__':
    cutct2img_test()
