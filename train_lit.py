import os
import torch
from matplotlib import pyplot as plt
import torch.utils
import torch.utils.data
from tooltip_feat_mask import LitTooltipNet
# from tooltip_resnet import LitTooltipNet
# from tooltip_unet import LitTooltipNet

import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from data import TooltipDataset, SimTooltipDataset, MixedTooltipDataset
import pytorch_lightning as pl

import argparse

"""
Training code of the TooltipNet
"""

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, lr_monitor, ModelCheckpoint

# Load the dataset
batch_size = 16
NUM_WORKERS = 8 #int(os.cpu_count() / 4)

data_type = "mixed" # "real" or "simulated" or "mixed" or "sim2real"
if data_type == "simulated":
    print("----------------Using simulated dataset----------------")
    mask_path = "/bigdata/SurgPose/tooltip/simulated_data"
    kps_file  = "/bigdata/SurgPose/tooltip/keypoints.json"
    dataset = SimTooltipDataset(mask_path=mask_path, kps_path=kps_file, mode="train")
    train_data, val_data = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
elif data_type == "sim2real":
    print("----------------Using simulated dataset training and real data for testing----------------")
    train_mask_path = "/bigdata/SurgPose/tooltip/simulated_data"
    val_mask_path   = "/bigdata/SurgPose/tooltip/eval"
    train_kps_file  = "/bigdata/SurgPose/tooltip/keypoints.json"
    val_kps_file    = "/bigdata/SurgPose/tooltip/keypoints_100003_left.yaml"
    train_data = SimTooltipDataset(mask_path=train_mask_path, kps_path=train_kps_file, mode="train")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_data = TooltipDataset(mask_path=val_mask_path, kps_path=val_kps_file, mode="val")
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
elif data_type == "mixed":
    print("----------------Using mixed dataset----------------")
    train_mask_path = "/bigdata/SurgPose/tooltip/mixed_data"
    val_mask_path   = "/bigdata/SurgPose/tooltip/new_eval"
    train_kps_file  = "/bigdata/SurgPose/tooltip/keypoints_mixed.json"
    val_kps_file    = "/bigdata/SurgPose/tooltip/new_kps_eval.json"

    train_data = SimTooltipDataset(train_mask_path, train_kps_file, mode="train")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)

    val_data = SimTooltipDataset(val_mask_path, val_kps_file, mode="val")
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=8)

# sanity check
# dataiter = iter(train_dataloader)
# images, labels = next(dataiter)

# Train the model
model = LitTooltipNet()

ckpt_callback = ModelCheckpoint(
    monitor='val_dist',
    mode='min',
    save_top_k=1,
    save_last=True,
    filename='best-ckpt-{epoch:02d}-{val_dist:.2f}',
)

trainer = pl.Trainer(
    max_epochs=100,
    accelerator="auto",
    devices="auto",
    logger=TensorBoardLogger(save_dir="logs/"),
    callbacks=[lr_monitor.LearningRateMonitor(logging_interval='step'), ckpt_callback],
)

trainer.fit(model, train_dataloader, val_dataloader)
