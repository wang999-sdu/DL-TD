import torch
from torch.utils.data import Dataset
import numpy as np


class My_Dataset(Dataset):
    """自定义数据集"""

    def __init__(self, image, label, transform=None):
        self.x = image
        self.y = label
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]
        label = self.y[idx]
        # 如果定义了transform方法，使用transform方法
        if self.transform:
            img = self.transform(img)
            # img, label = self.transform([img, label])
        # 因为上面我们已经把数据集处理好了生成了numpy形式，没必要处理了
        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        x, y = tuple(zip(*batch))

        x = torch.stack(x, dim=0)
        y = torch.as_tensor(y)
        return x, y
