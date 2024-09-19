import torch
import torchvision.models as models
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# 加载预训练的 ResNet50 模型
model = models.resnet50(pretrained=True)
model.eval()

# 定义 Grad-CAM 对象，指定要解释的层
cam_extractor = GradCAM(model, target_layer='layer4')

# 预处理函数
def preprocess_image(img_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(img_path).convert("RGB")
    return preprocess(image).unsqueeze(0)

# 加载和预处理图像
img_path = r'G:\data-CT\NNUNET_RESULT\NNUNET_RESULT\20240822_pic\17.nii.gz.jpg'
input_tensor = preprocess_image(img_path)

# 前向传播
output = model(input_tensor)

# 计算 Grad-CAM
activation_map = cam_extractor(output.squeeze(0).argmax().item(), output)
activation_map = torch.tensor(activation_map[0])

# 可视化 Grad-CAM 热力图
result = overlay_mask(to_pil_image(input_tensor.squeeze(0)), to_pil_image(activation_map, mode='F'), alpha=0.5)
plt.imshow(result)
plt.show()
