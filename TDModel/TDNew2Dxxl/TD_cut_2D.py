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
from warnings import simplefilter
from tqdm import tqdm  # 进度条库
import threading

simplefilter(action='ignore', category=FutureWarning)


"""
Get the pixel space of the ct nii file using pydicom and return mianji
"""
root_path = r"G:\data-CT"
train_path = root_path + r"\train.csv"
test_path = root_path + r"\test.csv"
img_path = root_path + r"\20240826xxl\image"
label_path = root_path + r"\20240826xxl\label"
# output_image_path = root_path + r"\20240826xxl\image_tumors_origin"
output_image_path = r"D:\xxl\image_tumors_origin"

train_list = pd.read_csv(train_path)


save_lock = threading.Lock()


def process_slice(name, img_path, label_path, output_image_path, progress_queue=None):
    ct_file_path = os.path.join(img_path, name + ".nii.gz")
    ct_label_file_path = os.path.join(label_path, name + ".nii.gz")
    ct_img_nib = nib.load(ct_file_path)
    ct_label_nib = nib.load(ct_label_file_path)
    ct_data = ct_img_nib.get_fdata()
    label_data = ct_label_nib.get_fdata()

    if len(ct_data.shape) == 5:
        ct_data = ct_data[:, :, :, 0, 0]
    if len(ct_data.shape) == 4:
        ct_data = ct_data[:, :, :, 0]

    header = ct_img_nib.header
    pixel_spacing = header.get_zooms()[:2]  # 通常前两个值为像素间距
    pixel_area = pixel_spacing[0] * pixel_spacing[1]
    tumor_area_threshold = 5

    for i in range(ct_data.shape[2]):
        # 获取当前切片的肿瘤区域
        tumor_slice = np.where(label_data[:, :, i] == 1, ct_data[:, :, i], 0)

        # 计算肿瘤区域的总面积
        tumor_area = np.sum(tumor_slice > 0) * pixel_area

        # 如果肿瘤面积大于阈值，则保存该切片
        if tumor_area >= tumor_area_threshold:
            # 调整切片大小为 240x240
            resized_slice = resize(tumor_slice, (240, 240), preserve_range=True)
            output_img_path = os.path.join(output_image_path, name + f"_{i}.png")
            # 保存图片
            with save_lock:
                plt.imshow(resized_slice, cmap='gray')
                plt.axis('off')
                plt.savefig(output_img_path, bbox_inches='tight', pad_inches=0)
    if progress_queue is not None:
        progress_queue.update(1)




names = train_list['患者编号'].tolist()

total_tasks = len(names)
progress_queue = tqdm(total=total_tasks, desc="Processing")
# for name in names:
    # process_slice(name, img_path, label_path, output_image_path, progress_queue)
# 使用多线程处理并显示进度

with concurrent.futures.ThreadPoolExecutor() as executor:
    for name in names:
        executor.submit(process_slice, name, img_path, label_path, output_image_path, progress_queue)
