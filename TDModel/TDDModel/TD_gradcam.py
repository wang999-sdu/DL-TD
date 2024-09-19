import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from torchvision.models import resnet50, resnet101
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os
from TDModel.TDDModel.model_train import ResNetClassifier
from tqdm import tqdm
import shutil


def te_gradcam():
    img_root_path = r"G:\data-CT\csv0828\image_train_val_0829"
    transform_3 = transforms.Grayscale(num_output_channels=3)
    transform_toTensor = transforms.ToTensor()
    targets = [ClassifierOutputTarget(1)]

    file_list = [f for f in os.listdir(img_root_path) if os.path.isfile(os.path.join(img_root_path, f))]

    for file_name in tqdm(file_list, desc="Processing files"):
        single_img_path = os.path.join(img_root_path, file_name)
        img = Image.open(single_img_path)
        img_array = np.array(transform_3(img)).astype(np.float32) / 255.0
        input_tensor = transform_toTensor(transform_3(img)).unsqueeze(0)
        data_save_path = rf".\img\gradcam\{file_name}"

        destination_file = os.path.join(data_save_path)
        shutil.copy(single_img_path, destination_file)

        model = ResNetClassifier(num_classes=2)
        target_layers = [model.resnet.layer4[-1]]
        with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)
            model_outputs = cam.outputs
            plt.imshow(visualization)
            plt.axis('off')
            plt.savefig(data_save_path + "_initial_.png", dpi=300, bbox_inches='tight')

        model.load_state_dict(torch.load("model/label1-ResNet101-checkpoint.pt"))

        with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)
            model_outputs = cam.outputs
            plt.imshow(visualization)
            plt.axis('off')
            plt.savefig(data_save_path + "_after_train_.png", dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    te_gradcam()

