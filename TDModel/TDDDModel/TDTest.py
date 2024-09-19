import numpy as np
import torch
import time
import os
from torch import nn
from scipy import ndimage
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from tqdm import tqdm
from utils.logger import log
from TDModel.TDModel.wtModel import generate_model


def wtTest(
    data_loader_test,
    model,
    sets,
):
    if torch.cuda.is_available():
        device = torch.device("cuda", sets.gpu_id)
    else:
        device = torch.device("cpu")

    with torch.no_grad():
        for x, label in tqdm(data_loader_test):
            x = x.float()
            x = x.to(device)
            label = label.to(device)


            # Forward pass
            logits = model(x)
            logits = logits.reshape([label.cpu().numpy().shape[0]])
            prob_out = nn.Sigmoid()(logits)
            pro_list = prob_out.detach().cpu().numpy()
