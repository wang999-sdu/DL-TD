from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

from networks.component.dataLoadMethod import download_data


def draw_oct(volume, type_volume='np', canal_first=False):
    if type_volume == 'np':
        if canal_first == False:
            print("taille du volume = %s (%s)" % (volume.shape, type_volume))
            slice_h_n, slice_d_n, slice_w_n = int(volume.shape[0] / 2), int(volume.shape[1] / 2), int(
                volume.shape[2] / 2)
            slice_h = volume[slice_h_n, :, :, :]
            slice_d = volume[:, slice_d_n, :, :]
            slice_w = volume[:, :, slice_w_n, :]
            slice_h = Image.fromarray(np.squeeze(slice_h))
            slice_d = Image.fromarray(np.squeeze(slice_d))
            slice_w = Image.fromarray(np.squeeze(slice_w))
            plt.figure(figsize=(21, 7))
            plt.subplot(1, 3, 1)
            plt.imshow(slice_h)
            plt.title(slice_h.size)
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(slice_d)
            plt.title(slice_d.size)
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(slice_w)
            plt.title(slice_w.size)
            plt.axis('off')
        if canal_first == True:
            print("taille du volume = %s (%s)" % (volume.shape, type_volume))
            slice_h_n, slice_d_n, slice_w_n = int(volume.shape[1] / 2), int(volume.shape[2] / 2), int(
                volume.shape[3] / 2)
            slice_h = volume[:, slice_h_n, :, :]
            slice_d = volume[:, :, slice_d_n, :]
            slice_w = volume[:, :, :, slice_w_n]
            slice_h = Image.fromarray(np.squeeze(slice_h))
            slice_d = Image.fromarray(np.squeeze(slice_d))
            slice_w = Image.fromarray(np.squeeze(slice_w))
            plt.figure(figsize=(21, 7))
            plt.subplot(1, 3, 1)
            plt.imshow(slice_h)
            plt.title(slice_h.size)
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(slice_d)
            plt.title(slice_d.size)
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(slice_w)
            plt.title(slice_w.size)
            plt.axis('off')

    if type_volume == 'tensor':
        if canal_first == False:
            print("taille du volume = %s (%s)" % (volume.shape, type_volume))
            slice_h_n, slice_d_n, slice_w_n = int(volume.shape[0] / 2), int(volume.shape[1] / 2), int(
                volume.shape[2] / 2)
            slice_h = volume[slice_h_n, :, :, :].numpy()
            slice_d = volume[:, slice_d_n, :, :].numpy()
            slice_w = volume[:, :, slice_w_n, :].numpy()
            slice_h = Image.fromarray(np.squeeze(slice_h))
            slice_d = Image.fromarray(np.squeeze(slice_d))
            slice_w = Image.fromarray(np.squeeze(slice_w))
            plt.figure(figsize=(21, 7))
            plt.subplot(1, 3, 1)
            plt.imshow(slice_h)
            plt.title(slice_h.size)
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(slice_d)
            plt.title(slice_d.size)
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(slice_w)
            plt.title(slice_w.size)
            plt.axis('off')
        if canal_first == True:
            slice_h_n, slice_d_n, slice_w_n = int(volume.shape[1] / 2), int(volume.shape[2] / 2), int(
                volume.shape[3] / 2)
            slice_h = volume[:, slice_h_n, :, :].numpy()
            slice_d = volume[:, :, slice_d_n, :].numpy()
            slice_w = volume[:, :, :, slice_w_n].numpy()
            slice_h = Image.fromarray(np.squeeze(slice_h))
            slice_d = Image.fromarray(np.squeeze(slice_d))
            slice_w = Image.fromarray(np.squeeze(slice_w))
            plt.figure(figsize=(21, 7))
            plt.subplot(1, 3, 1)
            plt.imshow(slice_h)
            plt.title(slice_h.size)
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(slice_d)
            plt.title(slice_d.size)
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(slice_w)
            plt.title(slice_w.size)
            plt.axis('off')
    plt.show()


# 3d数据可视化函数
def show_pic():
    train_dataset, val_dataset, test_dataset = download_data()
    x, y = train_dataset[0]
    print(x.shape, y)
    draw_oct(x * 500, type_volume='np', canal_first=True)
