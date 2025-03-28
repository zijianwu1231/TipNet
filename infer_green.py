import os
import torch
from matplotlib import pyplot as plt
from tooltip_resnet import LitTooltipNet

import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from data import TooltipDataset, GreenScreenInferDataset
import pytorch_lightning as pl

import argparse

"""
Reference code of the TooltipNet
"""

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, lr_monitor

from metrics import mean_euclidean_distance
import cv2
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the dataset
batch_size = 1
ckpt_path = "/home/zijianwu/Codes/TipNet/logs/lightning_logs/version_53/checkpoints/epoch=99-step=58300.ckpt"

infer_mask_path =   "/bigdata/SurgPose/tooltip/l_mask"

test_data = GreenScreenInferDataset(mask_path=infer_mask_path)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

# sanity check
# dataiter = iter(train_dataloader)
# images, labels = next(dataiter)

# Load the model checkpoint
model = LitTooltipNet.load_from_checkpoint(ckpt_path)
model.eval()

# predict directory
os.makedirs("result_vis_green", exist_ok=True)

# Test the model
for idx, mask in tqdm(enumerate(test_dataloader)):
    mask = mask.to(DEVICE)
    pred_kps = model(mask)
    print(f"Predicted keypoints: {pred_kps}")
    # print(f"Mean euclidean distance: {mean_euclidean_distance(pred_kps, kps)}")
    print("\n")
    
    # Visualize the keypoints
    mask = mask.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    pred_kps = pred_kps.cpu().detach().numpy().squeeze(0)

    for kp in pred_kps:
        mask = cv2.circle(mask, (int(kp[0]), int(kp[1])), 5, (255, 255, 255), -1)

    # figure
    # no margins
    plt.axis('off')
    plt.imshow(mask)

    # save the visualization
    plt.savefig(f"result_vis_green/{idx}.png", bbox_inches="tight", pad_inches=0)