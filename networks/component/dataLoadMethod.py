import medmnist
from medmnist import INFO, Evaluator
import numpy as np

# 数据处理
class Transform3D:

    def __init__(self, mul=None):
        self.mul = mul

    def __call__(self, voxel):
        if self.mul == '0.5':
            voxel = voxel * 0.5
        elif self.mul == 'random':
            voxel = voxel * np.random.uniform()
        return voxel.astype(np.float32)


def download_data():
    print('==> Preparing data...')
    train_transform = Transform3D(mul='random')
    eval_transform = Transform3D(mul='0.5')

    data_flag = 'organmnist3d'  # Multi-Class (11)
    download = True

    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])



    # load the data
    train_dataset = DataClass(split='train', transform=train_transform, download=download)
    val_dataset = DataClass(split='val', transform=eval_transform, download=download)
    test_dataset = DataClass(split='test', transform=eval_transform, download=download)
    return train_dataset, val_dataset, test_dataset