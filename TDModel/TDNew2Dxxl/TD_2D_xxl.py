import os
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import argparse
import pandas as pd
import json

dataset_root = r"G:\data-CT"

label_file = os.path.join(dataset_root, 'train.csv')
source_dir = r'G:\data-CT\NNUNET_RESULT\NNUNET_RESULT\20240822_pic'
destination_dir = r'G:\data-task-1-xxl\xxl'
xxl_data = pd.read_csv(label_file)

train_df, test_df = train_test_split(xxl_data, test_size=0.25, random_state=42)

train_output_dict = list()
test_output_dict = list()
for name, label in zip(train_df['患者编号'], train_df['label1']):
    source_name = name + ".nii.gz.jpg"
    source_path = os.path.join(source_dir, source_name)
    path_dict = {}
    path_dict["image_path"] = source_path
    path_dict["target"] = 0 if label == "N" else 1
    train_output_dict.append(path_dict)

for name, label in zip(test_df['患者编号'], test_df['label1']):
    source_name = name + ".nii.gz.jpg"
    source_path = os.path.join(source_dir, source_name)
    path_dict = {}
    path_dict["image_path"] = source_path
    path_dict["target"] = 0 if label == "N" else 1
    test_output_dict.append(path_dict)

json.dump(train_output_dict, open(r"./data/train_output_dict.json", 'w'))
json.dump(test_output_dict, open(r"./data/test_output_dict.json", 'w'))
