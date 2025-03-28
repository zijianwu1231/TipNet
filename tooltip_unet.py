import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet50_Weights
from torch.autograd import Variable
import numpy as np
import torchvision.models as models
import argparse
import pytorch_lightning as pl
from torchmetrics import Accuracy
from metrics import mean_euclidean_distance
import tensorboard
    
class DoubleConv(nn.Module):
    """(Conv => BatchNorm => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet3Layer(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        
        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(F.max_pool2d(x1, 2))
        x3 = self.enc3(F.max_pool2d(x2, 2))

        # Bottleneck
        x4 = self.bottleneck(F.max_pool2d(x3, 2))

        # Decoder with skip connections
        x = self.up3(x4)
        x = self.dec3(torch.cat([x, x3], dim=1))
        
        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))
        
        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))

        return self.out_conv(x)

class SpatialSoftArgmax(nn.Module):
    """
    The spatial softmax of each feature
    map is used to compute a weighted mean of the pixel
    locations, effectively performing a soft arg-max
    over the feature dimension.

    """

    def __init__(self, normalize=True):
        """Constructor.
        Args:
            normalize (bool): Whether to use normalized
                image coordinates, i.e. coordinates in
                the range `[-1, 1]`.
        """
        super().__init__()

        self.normalize = normalize

    def _coord_grid(self, h, w, device):
        if self.normalize:
            return torch.stack(
                torch.meshgrid(
                    torch.linspace(-1, 1, h, device=device),
                    torch.linspace(-1, 1, w, device=device),
                    indexing='ij',
                )
            )
        return torch.stack(
            torch.meshgrid(
                torch.arange(0, h, device=device),
                torch.arange(0, w, device=device),
                indexing='ij',
            )
        )

    def forward(self, x):
        assert x.ndim == 4, "Expecting a tensor of shape (B, C, H, W)."

        # compute a spatial softmax over the input:
        # given an input of shape (B, C, H, W),
        # reshape it to (B*C, H*W) then apply
        # the softmax operator over the last dimension
        b, c, h, w = x.shape
        softmax = F.softmax(x.view(-1, h * w), dim=-1)

        # create a meshgrid of pixel coordinates
        # both in the x and y axes
        yc, xc = self._coord_grid(h, w, x.device)

        # element-wise multiply the x and y coordinates
        # with the softmax, then sum over the h*w dimension
        # this effectively computes the weighted mean of x
        # and y locations
        x_mean = (softmax * xc.flatten()).sum(dim=1, keepdims=True)
        y_mean = (softmax * yc.flatten()).sum(dim=1, keepdims=True)

        # concatenate and reshape the result
        # to (B, C, 2) where for every feature
        # we have the expected x and y pixel
        # locations
        return torch.cat([x_mean, y_mean], dim=1).view(-1, c, 2)

class TipUNet(nn.Module):
    def __init__(self, n_kp, w, h, lim=[-1., 1., -1., 1.], use_gpu=True):
        super(TipUNet, self).__init__()

        self.lim = lim

        self.width = w
        self.height = h
        k = n_kp

        if use_gpu:
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.backbone = UNet3Layer(in_channels=3, out_channels=n_kp)
        self.spatialsoftargmax = SpatialSoftArgmax()

    def forward(self, img):
        input_shape = img.shape[-2:]
        img = img.float()
        unet_out = self.backbone(img)  # unet_out: (B, k, H, W)
        keypoints = self.spatialsoftargmax(unet_out)

        # mapping back to original resolution from [-1,1]
        offset = torch.tensor([self.lim[0], self.lim[2]], device = unet_out.device)
        scale = torch.tensor([self.width // 2, self.height // 2], device = unet_out.device)
        keypoints = keypoints - offset
        keypoints = keypoints * scale

        return keypoints
    
class LitTooltipNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = TipUNet(n_kp=2, w=640, h=512)
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        mask, kps = batch
        logits = self.model(mask)
        loss = self.loss_fn(logits, kps)
        self.log('train_loss', loss)
        dist = mean_euclidean_distance(logits.cpu().detach().numpy(), kps.cpu().detach().numpy())
        self.log('train_dist', dist)
        return loss
    
    def validation_step(self, batch, batch_idx):
        mask, kps = batch
        logits = self.model(mask)
        loss_val = self.loss_fn(logits, kps)
        self.log('val_loss', loss_val)
        dist = mean_euclidean_distance(logits.cpu().detach().numpy(), kps.cpu().detach().numpy())
        self.log('val_dist', dist)

        # print(acc)
        return loss_val

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [lr_scheduler]

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--n_kp", type=int, default=2, help="Number of keypoints")
    argparse.add_argument("--width", type=int, default=640, help="Image width")
    argparse.add_argument("--height", type=int, default=512, help="Image height")
    args = argparse.parse_args()

    tip_unet = UNet3Layer(in_channels=3, out_channels=2)
    print(tip_unet)

    input_tensor = torch.randn(8, 3, 512, 640)
    output_tensor = tip_unet(input_tensor)
    print(output_tensor.shape)