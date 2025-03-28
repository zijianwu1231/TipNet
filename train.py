import torch
import torchvision
from torch.utils.data import DataLoader
from tooltip_resnet import TooltipNet
from data import TooltipDataset

import argparse
import matplotlib.pyplot as plt
import numpy as np
from metrics import mean_euclidean_distance
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", "-bs", type=int, default=8, help="Batch size")
parser.add_argument("--train_dir", "-td", type=str, default="/bigdata/SurgPose/tooltip/train", help="Path to the training dataset")
parser.add_argument("--val_dir", "-vd", type=str, default="/bigdata/SurgPose/tooltip/eval", help="Path to the validation dataset")
parser.add_argument("--train_kps", "-tk", type=str, default="/bigdata/SurgPose/tooltip/keypoints_100002_left.yaml", help="Path to the training keypoints file")
parser.add_argument("--val_kps", "-vk", type=str, default="/bigdata/SurgPose/tooltip/keypoints_100003_left.yaml", help="Path to the validation keypoints file")
parser.add_argument("--n_kp", type=int, default=4, help="Number of keypoints")
parser.add_argument("--width", type=int, default=640, help="Image width")
parser.add_argument("--height", type=int, default=512, help="Image height")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")

args = parser.parse_args()

batch_size = args.batch_size
init_lr = args.lr
n_kp = args.n_kp
width = args.width
height = args.height
epoch_num = args.epochs

def imshow(img):
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Load Dataset
train_mask_dir = args.train_dir
train_kps_file = args.train_kps
val_mask_dir   = args.val_dir
val_kps_file   = args.val_kps

train_data = TooltipDataset(mask_path=train_mask_dir, kps_path=train_kps_file)
val_data = TooltipDataset(mask_path=val_mask_dir, kps_path=val_kps_file)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16)
val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=16)

# dataiter = iter(train_loader) # sanity check
# images, labels = next(dataiter)
# imshow(torchvision.utils.make_grid(images))

# Load Model
model = TooltipNet(n_kp=n_kp, w=width, h=height)

# initialize optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
loss_fn = torch.nn.L1Loss() #torch.nn.CrossEntropyLoss()

# Train the model

loss_list_tr = []
loss_list_val = []
acc_list_tr = []
acc_list_val = []

for epoch in range(epoch_num):
    model.train()
    epoch_loss_tr = 0
    epoch_acc_tr = 0
    for idx, batch in tqdm(enumerate(train_loader)):
        mask, kps = batch
        optimizer.zero_grad()
        logits = model(mask)
        loss_tr = loss_fn(logits, kps)
        loss_tr.backward()
        optimizer.step()
        epoch_loss_tr += loss_tr.item()
        acc_tr = mean_euclidean_distance(logits.cpu().detach().numpy(), kps.cpu().detach().numpy())
        epoch_acc_tr += acc_tr

    print(f"Epoch: {epoch}, Loss: {epoch_loss_tr}, Acc: {epoch_acc_tr}")

    model.eval()
    epoch_loss_val = 0
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(val_loader)):
            mask, kps = batch
            logits = model(mask)
            loss_val = loss_fn(logits, kps)
            epoch_loss_val += loss_val.item()
            acc_val = mean_euclidean_distance(logits.cpu().detach().numpy(), kps.cpu().detach().numpy())
            epoch_acc_val += acc_val

    print(f"Epoch: {epoch}, Validation Loss: {epoch_loss_val}, Validation Acc: {epoch_acc_val}")

    loss_list_tr.append(epoch_loss_tr)
    loss_list_val.append(epoch_loss_val)
    acc_list_tr.append(epoch_acc_tr)
    acc_list_val.append(epoch_acc_val)

    # Plot and save the train loss, validation loss, train acc, validation acc in a 2 rows 2 columns grid plot
    fig = plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(loss_list_tr)

    plt.subplot(2, 2, 2)
    plt.plot(loss_list_val)

    plt.subplot(2, 2, 3)
    plt.plot(acc_list_tr)

    plt.subplot(2, 2, 4)
    plt.plot(acc_list_val)

    plt.savefig("loss_acc_plot.png")



# Save the model
# torch.save(model.state_dict(), "tooltip_model.pth")