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
import cv2

class KeypointUpSample(nn.Module):
    def __init__(self, in_channels, num_keypoints, feature_channels=256):
        super().__init__()
        input_features = in_channels
        deconv_kernel = 4
        self.kps_score_lowres = nn.ConvTranspose2d(
            input_features,
            int(input_features/32),
            deconv_kernel,
            stride=2,
            padding=deconv_kernel // 2 - 1,
        )

        self.convtrans = nn.ConvTranspose2d(int(input_features/32)+feature_channels, num_keypoints, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1,)
        self.conv = nn.Conv2d(int(input_features/32)+feature_channels, num_keypoints, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)

        self.bn = nn.BatchNorm2d(int(input_features/32))
        self.relu = nn.ReLU()
        
        nn.init.kaiming_normal_(self.kps_score_lowres.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.kps_score_lowres.bias, 0)
        #nn.init.uniform_(self.kps_score_lowres.weight)
        #nn.init.uniform_(self.kps_score_lowres.bias)
        self.up_scale = 1
        self.out_channels = num_keypoints

    def forward(self, x, feat):
        x = self.kps_score_lowres(x) # (B, 1024, H//4, W//4)
        x = self.bn(x)
        x = self.relu(x)

        # Concatenate the low-res feature map with the high-res feature map
        x = torch.cat([x, feat], dim=1) # (B, 1280, H//4, W//4)

        # x = self.conv(x)        # (B, 2, H//4, W//4)
        x = self.convtrans(x)   # (B, 2, H//2, W//2)

        return torch.nn.functional.interpolate(
            x, scale_factor=float(self.up_scale), mode="bilinear", align_corners=False, recompute_scale_factor=False
        )

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

    def forward(self, x, part_mask=None):
        assert x.ndim == 4, "Expecting a tensor of shape (B, C, H, W)."

        # compute a spatial softmax over the input:
        # given an input of shape (B, C, H, W),
        # reshape it to (B*C, H*W) then apply
        # the softmax operator over the last dimension
        b, c, h, w = x.shape
        softmax = F.softmax(x.view(-1, h * w), dim=-1)

        # apply part_mask to the softmax
        if part_mask is not None:
            part_mask = part_mask.view(-1, h * w)
            # concatenate 2 part_mask to match the softmax shape
            part_mask = torch.cat([part_mask, part_mask], dim=0)
            softmax = softmax * part_mask

            # argmax along the h*w dimension and get the idx

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

class TooltipNet(nn.Module):
    def __init__(self, n_kp, w, h, lim=[-1., 1., -1., 1.], use_gpu=True):
        super(TooltipNet, self).__init__()

        self.lim = lim

        self.width = w
        self.height = h
        k = n_kp

        if use_gpu:
            self.device = "cuda"
        else:
            self.device = "cpu"

        deeplabv3_resnet50 = models.segmentation.deeplabv3_resnet50(pretrained=True, weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        # deeplabv3_resnet50.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1)) # Change final layer to 2 classes

        self.backbone = deeplabv3_resnet50.backbone #torch.nn.Sequential(list(deeplabv3_resnet50.children())[0])

        self.read_out = KeypointUpSample(2048, k)

        self.spatialsoftargmax = SpatialSoftArgmax()

    def forward(self, img):
        input_shape = img.shape[-2:]
        img = img.float()

        # get gripper mask
        # img is a 3 channel mask, red - gripper, green - tooltip, blue - shaft
        # how to keep the blue [255, 0, 0] only?
        part_mask = img[:, 2, :, :] > 0
        part_mask = part_mask.unsqueeze(1) # (B, 1, H, W)
        part_mask = F.interpolate(part_mask.float(), size=(self.height//2, self.width//2), mode='nearest') # (B, 1, H//2, W//2)
        # mask = mask.float()
        # img = mask
        # from matplotlib import pyplot as plt
        # plt.imshow(part_mask[0, 0, :, :].cpu().detach().numpy())
        # plt.savefig('part_mask.png')
        # breakpoint()

        # resnet_out, feat = self.backbone(img)['out']  # (B, 2048, H//8, W//8) feat: (B, 256, H//4, W//4)
        resnet_out = self.backbone(img)  # resnet_out: (B, 2048, H//8, W//8)

        resnet_out, feat = resnet_out['out'], resnet_out['aux']  # resnet_out: (B, 2048, H//8, W//8), feat: (B, 256, H//4, W//4)

        # keypoint prediction branch
        # heatmap = self.read_out(resnet_out, feat) # (B, k, H//4, W//4)
        heatmap = self.read_out(resnet_out, feat) # (B, k, H//2, W//2)

        # apply part_mask to the heatmap

        # breakpoint()
        # keypoints = self.spatialsoftargmax(heatmap, part_mask)
        heatmap = heatmap * part_mask
        keypoints = self.spatialsoftargmax(heatmap)

        # mapping back to original resolution from [-1,1]
        offset = torch.tensor([self.lim[0], self.lim[2]], device = resnet_out.device)
        scale = torch.tensor([self.width // 2, self.height // 2], device = resnet_out.device)
        keypoints = keypoints - offset
        keypoints = keypoints * scale

        return keypoints
    
class LitTooltipNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = TooltipNet(n_kp=2, w=640, h=512)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [lr_scheduler]

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--n_kp", type=int, default=2, help="Number of keypoints")
    argparse.add_argument("--width", type=int, default=640, help="Image width")
    argparse.add_argument("--height", type=int, default=512, help="Image height")
    args = argparse.parse_args()

    # tipnet = TooltipNet(args.n_kps, args.width, args.height)
    model = LitTooltipNet()

    print(model)