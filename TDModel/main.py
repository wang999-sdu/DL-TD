import nibabel as nib
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from TDModel.wtData.wtData1 import wt_create_data_index, wt_data_test, wt_get_data_label
from TDModel.wtData.wtDataset import WtThymusDataset
from TDModel.wtModel.wtModel import generate_model
from TDModel.wtModel.wtTrain import wtTrain
from TDModel.wtSet.wtSetting import parse_opts
from sklearn.model_selection import train_test_split
import nibabel
import pandas as pd

summaryWriter = SummaryWriter("wtCommons/")
batch_size = 256


def wt_init_setOpts():
    sets = parse_opts()
    sets.img_list = './wtData/train.txt'
    sets.n_epochs = 50
    sets.ct_data_root = 'G:\\data-CT\\NNUNET_RESULT\\NNUNET_RESULT'
    sets.name_list = 'E:\\workspace_python311\\project-thymoma\\wtlizzz-core\\wtliModel\\wtData\\data\\label1\\testList.txt'
    sets.label_path = 'E:\\workspace_python311\\project-thymoma\\wtlizzz-core\\wtliModel\\wtData\\label\\label1.csv'
    sets.label_flag = 'L'
    sets.num_workers = 0
    sets.model_depth = 10
    sets.resnet_shortcut = 'A'
    sets.pin_memory = True
    sets.model_type = 'resnet'
    sets.input_W = 224
    sets.input_H = 224
    sets.input_D = 224
    sets.no_cuda = False
    sets.gpu_id = 0
    sets.pretrain_path = './wtWeight/50.pth'
    # sets.pretrain_path = '../model/pretrain/resnet_50.pth'
    sets.nb_class = 1
    return sets


# 创建txt图像所引文件，只需要运行一次
def wt_data_loader_1():
    image_path = "G:\\data-CT\\alldata\\imagesTr"
    label_path = "G:\\data-CT\\alldata\\labelsTr"
    out_path = "./wtData/test.txt"
    wt_create_data_index(image_path, label_path, out_path)


def wt_data_loader_2(sets):
    sets.mask_path = "./wtData/trainLabel.txt"
    dataset_training = WtThymusDataset(sets)
    data_loader_train = DataLoader(dataset_training, batch_size=sets.batch_size, shuffle=True,
                                   num_workers=sets.num_workers,
                                   pin_memory=sets.pin_memory)
    sets.img_list = './wtData/val.txt'
    sets.mask_path = "./wtData/valLabel.txt"
    dataset_val = WtThymusDataset(sets)
    data_loader_val = DataLoader(dataset_val, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers,
                                 pin_memory=sets.pin_memory)

    return data_loader_train, data_loader_val


def wt_init_train(sets, data_loader_train, data_loader_val):
    model = generate_model(sets)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0])).cuda()  # 分类不均衡
    # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    # training
    wtTrain(data_loader_train, data_loader_val, summaryWriter, model, optimizer, scheduler, criterion,
            total_epochs=sets.n_epochs,
            save_interval=sets.save_intervals,
            save_folder=sets.save_folder, sets=sets)


def wt_main():
    sets = wt_init_setOpts()
    data_loader_train, data_loader_val = wt_data_loader_2(sets)
    wt_init_train(sets, data_loader_train, data_loader_val)


def wt_nature():
    from scipy import ndimage
    import matplotlib.pyplot as plt
    import numpy as np
    img = nibabel.load("G:\\data-CT\\NNUNET_RESULT\\NNUNET_RESULT\\image\\KF10.nii.gz")
    # 获得图像数据形状
    print("图像数据形状为{}".format(img.shape))

    data = np.asanyarray(img.dataobj)
    print("浮点图像数据类型为{}".format(type(data)))

    def show_slices(slices):
        """ 显示一行图像切片 """
        fig, axes = plt.subplots(1, len(slices))
        for i, slice in enumerate(slices):
            axes[i].imshow(slice.T, cmap="gray", origin="lower")
        plt.show()

    item_data_1 = data[:, :, :, 0, 0]
    item_data_2 = data[:, :, :, 0, 1]

    # 获得三个维度的切片
    slice_0 = item_data_1[120, :, :]
    slice_1 = item_data_1[:, 120, :]
    slice_2 = item_data_1[:, :, 40]
    slice_3 = item_data_2[120, :, :]
    slice_4 = item_data_2[:, 120, :]
    slice_5 = item_data_2[:, :, 40]
    show_slices([slice_0, slice_1, slice_2])
    show_slices([slice_3, slice_4, slice_5])
    plt.suptitle("Center slices for brain tumor image")

    [depth, height, width] = data.shape
    scale = [224 * 1.0 / depth, 224 * 1.0 / height, 224 * 1.0 / width]
    data2 = ndimage.interpolation.zoom(data, scale, order=0)
    return data2


def wt_test(init_set, model):
    from tqdm import tqdm
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")

    dataset_val = WtThymusDataset(init_set)
    data_loader_test = DataLoader(dataset_val, batch_size=init_set.batch_size, shuffle=True,
                                  num_workers=init_set.num_workers,
                                  pin_memory=init_set.pin_memory)
    res_table = {'predict_value': [], 'prob_out': [], 'predict_stand': [], 'label': [], 'result_boolean': []}
    res_df = pd.DataFrame(res_table)

    with torch.no_grad():
        for x, label in tqdm(data_loader_test):
            x = x.float()
            x = x.to(device)
            label = label.to(device)
            logits = model(x)
            # logits = logits.reshape([label.cpu().numpy().shape[0]])
            logits = logits.cpu().item()
            prob_out = nn.Sigmoid()(logits)
            res = logits.item()
            label_value = label.item()
            if res > 0:
                predict_stand = 1
            else:
                predict_stand = 0
            prob_out_value = prob_out.item()

            result_boolean = predict_stand == label_value
            print(
                f"res: {res}, prob_out: {prob_out_value}, predict_stand: {round(prob_out_value)}, label: {label_value},result_boolean: {result_boolean}")
            res_df.loc[len(res_df)] = [res, prob_out_value, round(prob_out_value), label_value, result_boolean]
    res_df.to_csv("wt_test.csv", index=False)


if __name__ == '__main__':
    init_set = wt_init_setOpts()
    model = generate_model(init_set)
    wt_test(init_set, model)
