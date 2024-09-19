import nibabel as nb
import numpy as np
from PIL import Image
from nibabel.viewers import OrthoSlicer3D
import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt

def ct_pic_handle():
    img = nb.load("G:\\data-CT\\alldata\\imagesTr\\11_0000_0000.nii.gz")  # 读取nii格式文件
    img_affine = img.affine
    data = np.asanyarray(img.dataobj)
    nb.Nifti1Image(data, img_affine).to_filename(".\\img_affine2.nii.gz")
    OrthoSlicer3D(data.transpose(1, 2, 0)).show()


def get_ct_pic():
    image_path = "G:\\data-CT\\alldata\\imagesTr\\11_0000_0000.nii.gz"
    mask_path = "G:\\data-CT\\alldata\\labelsTr\\11_0000.nii.gz"
    img = nb.load(image_path)
    img_affine = img.affine
    mask = nb.load(mask_path)
    data = np.asanyarray(img.dataobj)
    label = np.asanyarray(mask.dataobj)

    # zero_value = data[0, 0, 0]
    zero_value = label[0, 0, 0]
    non_zeros_idx = np.where(label != zero_value)

    [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
    [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)
    data = data[min_z:max_z, min_h:max_h, min_w:max_w]

    # nb.save(data, ".\\img_affine1.nii.gz")
    nb.Nifti1Image(data, img_affine).to_filename(".\\img_affine1.nii.gz")

    if label is not None:
        return data[min_z:max_z, min_h:max_h, min_w:max_w], label[min_z:max_z, min_h:max_h, min_w:max_w]
    else:
        return data[min_z:max_z, min_h:max_h, min_w:max_w]


def data_handle():
    '''有的ct文件shape是5维的data.shape:(53, 62, 10, 1, 2)，将5维转化成3维'''
    # img = nb.load("C:\\Users\\61708\\Desktop\\123123\\KF6.nii.gz")  # 读取nii格式文件
    img_list = os.listdir("C:\\Users\\61708\\Desktop\\123123")
    for img in img_list:
        label = nb.load("C:\\Users\\61708\\Desktop\\123123\\" + img)
        data = np.asanyarray(label.dataobj)
        print(f"img:{img},  data.shape:{data.shape}")

    # [z, y, x, a, b] = data.shape
    # new_data = np.reshape(data, [z, y, x])
    # nb.Nifti1Image(data, img_affine).to_filename("C:\\Users\\61708\\Desktop\\123123\\KF6-1.nii.gz")


def wt_test_gpt():
    dicom_file = pydicom.dcmread('path_to_your_ct_image.dcm')
    ct_image = dicom_file.pixel_array
    plt.imshow(ct_image, cmap='gray')
    plt.show()


if __name__ == '__main__':
    wt_test_gpt()
