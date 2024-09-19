# 读取数据
import json
import os
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image


def data_load():
    label_path = "G:\\data-CT\\NNUNET_RESULT\\NNUNET_RESULT\\LABEL.csv"
    root_path = r"G:\data-CT\png224"
    data_scan_paths = [
        x
        for x in os.listdir(root_path)
    ]
    train_set, test_set = train_test_split(data_scan_paths, test_size=0.2, random_state=42)
    mask_list = pd.read_csv(label_path)
    mask_list["患者编号"] = mask_list["患者编号"].str.lower()

    num = 0
    train_scans = []
    train_y_set = []
    for train_name in train_set:
        # print(f"num in train_set: {num}:{len(train_set)}")
        all_path = os.path.join(root_path, train_name)
        img = Image.open(all_path)
        if img.mode != 'RGB':
            img = img.convert("RGB")
        train_scans.append(img)
        mask = mask_list.loc[mask_list["患者编号"] == train_name[: train_name.find(".")]].label1.to_string(index=False)
        print(f"train_name:{train_name}  num in train_set: {num}:{len(train_set)} | mask: {mask}")
        train_y_set.append(0 if mask == "N" else 1)
        num += 1
        # if num == 2:
        #     break

    # train_scans_np = np.asarray(train_scans)
    # train_y_set_np = np.asarray(train_y_set)

    num = 0
    test_scans = []
    test_y_set = []
    for test_name in test_set:
        all_path = os.path.join(root_path, test_name)
        img = Image.open(all_path)
        if img.mode != 'RGB':
            img = img.convert("RGB")
        test_scans.append(img)
        mask = mask_list.loc[mask_list["患者编号"] == test_name[: test_name.find(".")]].label1.to_string(index=False)
        print(f"test_name:{test_name}  num in train_set: {num}:{len(test_set)} | mask:{mask}")
        test_y_set.append(0 if mask == "N" else 1)
        num += 1
        # if num == 2:
        #     break

    # test_scans_np = np.asarray(test_scans)
    # test_y_set_np = np.asarray(test_y_set)

    return train_scans, train_y_set, test_scans, test_y_set


def read_split_data(data_root, save_dir, val_rate=0.2, plot_iamge=False):
    # 随机种子，确保每次结果可复现
    random.seed(0)

    # 遍历data_root下的文件夹，一个文件夹对应一个类别
    classes = [cla for cla in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, cla))]
    # 排序
    classes.sort()

    # 分类任务，需要索引，所以创建类别名称以及对应的索引
    class_indices = dict((k, v) for v, k in enumerate(classes))
    # 将类别以及索引写入json文件
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open(save_dir + '/class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # 存放训练集图片路径
    train_images_path = []
    # 存放验证集图片路径
    val_images_path = []
    # 存放训练集标签
    train_labels = []
    # 存放验证集标签
    val_labels = []
    # 存放每个类别的图片数量
    every_class_num = []
    # 图片文件所能支持的格式
    supported = ['.jpg', '.jpeg', '.png']

    # 将图片路径保存至txt
    train_txt = open(save_dir + '/train.txt', 'w')
    val_txt = open(save_dir + '/val.txt', 'w')

    # 遍历每一个标签文件夹，读取图片
    for cla in tqdm(classes):
        cla_path = os.path.join(data_root, cla)
        # 遍历获取对应文件夹下的所有图片
        images = [os.path.join(cla_path, i) for i in os.listdir(cla_path) if
                  os.path.splitext(i)[-1].lower() in supported]
        # 获取图片标签
        image_class = class_indices[cla]
        # 记录该类别的图片数量
        every_class_num.append(len(images))
        # 按比例划分训练集和验证集
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for image_path in images:
            if image_path in val_path:
                val_txt.write(image_path + '\n')
                val_images_path.append(image_path)
                val_labels.append(image_class)
            else:
                train_txt.write(image_path + '\n')
                train_images_path.append(image_path)
                train_labels.append(image_class)

    train_txt.close()
    val_txt.close()

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    if plot_iamge:
        plt.bar(range(len(classes)), every_class_num, align='center')
        plt.xticks(range(len(classes)), classes)
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        plt.xlabel('image classes')
        plt.ylabel('number of image')
        plt.title('class distribution')
        plt.savefig(os.path.join(save_dir, 'classes.jpg'))
    return train_images_path, val_images_path, train_labels, val_labels, every_class_num


if __name__ == '__main__':
    read_split_data(data_root=r'E:\dataset\flow_data\train', save_dir='./', plot_iamge=True)
