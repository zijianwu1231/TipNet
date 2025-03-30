import os
import torch
from torch.utils.data import Dataset
import albumentations as A 
import numpy as np
import torch.utils.data
import yaml
import cv2
import matplotlib.pyplot as plt
import torchvision
import json

"""
Dataloader of the tooltip dataset
"""
def imshow(img):
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class TooltipDataset():
    def __init__(self, mask_path, kps_path, mode, transform=None):
        self.mask_path = mask_path
        self.kps_file = kps_path
        self.mode = mode
        self.transform = transform
        self.is_inst = False
        self.single_obj = True

        self.mask_list = sorted(os.listdir(mask_path))
        self.frame_list = [int(mask.split('.')[0][5:]) for mask in self.mask_list]

        kps = yaml.load(open(kps_path, "r"), Loader=yaml.FullLoader)
        self.tooltip_idx = [4, 5, 11, 12] # psm1 - 4,5; psm3 - 11,12

        self.kps_tip = []
        for frame in self.frame_list:
            kps_frame_full = kps[frame]
            kps_frame_tip = []
            for idx in self.tooltip_idx:
                # kps_frame_full[idx] = [int(kps_frame_full[idx][0]), int(kps_frame_full[idx][1])]
                kps_frame_tip.append(tuple(kps_frame_full[idx]))
                # kps_frame_tip.append(kps_frame_full[idx])
            self.kps_tip.append(kps_frame_tip)
        
    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, idx):
        mask = cv2.imread(os.path.join(self.mask_path, self.mask_list[idx]))
        kps = self.kps_tip[idx]

        # transform
        if self.mode == "train":
            self.transform = TooltipTransform()
            mask, kps = self.transform(mask, kps)
        if self.mode == "val":
            self.transform = TooltipInferTransform()
            mask, kps = self.transform(mask, kps)

        # generate instance-level mask label
        mask_inst = np.where(np.any(mask > 0, axis=-1), 255, 0).astype(np.uint8)

        # generate single instance label

        if self.is_inst and self.single_obj:
            # connected component analysis and contour extraction
            n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_inst, connectivity=8)

            # randomly choose one instance
            inst_idx = np.random.randint(1, n_labels)
            mask_inst_single = np.where(labels == inst_idx, 255, 0).astype(np.uint8)
            mask_vis = np.where(labels == inst_idx, 255, 0).astype(np.uint8)
            mask_vis = cv2.merge((mask_vis, mask_vis, mask_vis))

            # calculate the center of this instance
            center = centroids[inst_idx]

            # generate corresponding keypoints
            # find 2 closest points to the center
            dist_list = []
            kps_single = []
            for kp in kps:
                dist = np.linalg.norm(np.array(kp) - center)
                dist_list.append(dist)

            dist_list = np.array(dist_list)
            dist_list_sorted = np.argsort(dist_list)

            kps_single.append(kps[dist_list_sorted[0]])
            kps_single.append(kps[dist_list_sorted[1]])

            # generate corresponding keypoints
            # kps_single = []
            # for kp in kps:
            #     if mask_inst_single[int(kp[1]), int(kp[0])] == 1:
            #         kps_single.append(kp)
            #         # mask_vis = cv2.circle(mask_vis, (int(kp[0]), int(kp[1])), 5, (255, 0, 0), -1)
            # kps_single = np.array(kps_single)

            # plt.imshow(mask_vis)
            # plt.show()
            # breakpoint()

            mask = cv2.merge((mask_inst_single, mask_inst_single, mask_inst_single))
            kps = kps_single
        
        elif self.is_inst and not self.single_obj:
            mask = cv2.merge((mask_inst, mask_inst, mask_inst))

        elif not self.is_inst and self.single_obj:
            n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_inst, connectivity=8)
            inst_idx = np.random.randint(1, n_labels)
            mask_inst_single = np.where(labels == inst_idx, 255, 0).astype(np.uint8)
            mask_inst_bool = np.where(labels == inst_idx, True, False)
            mask_vis = np.where(labels == inst_idx, 255, 0).astype(np.uint8)
            mask_vis = cv2.merge((mask_vis, mask_vis, mask_vis))
            center = centroids[inst_idx]
            
            dist_list = []
            kps_single = []
            for kp in kps:
                dist = np.linalg.norm(np.array(kp) - center)
                dist_list.append(dist)

            dist_list = np.array(dist_list)
            dist_list_sorted = np.argsort(dist_list)

            kps_single.append(kps[dist_list_sorted[0]])
            kps_single.append(kps[dist_list_sorted[1]])

            kps = kps_single

            # given the instance mask, select the part-level semantic mask
            mask_inst_bool = np.stack([mask_inst_bool] * 3, axis=-1)
            mask = mask * mask_inst_bool

        # draw keypoints on mask
        # for kp in kps:
        #     mask = cv2.circle(mask, (int(kp[0]), int(kp[1])), 5, (255, 0, 0), -1)
        # plt.imshow(mask)
        # plt.show()

        # convert to tensor
        mask = torch.from_numpy(mask).permute(2, 0, 1) # (C, H, W)
        kps = torch.tensor(kps)
        
        # assert kps.shape[0] == 2, f"{self.mask_list[idx]}, {kps.shape} keypoints should be in shape (2, n_kps)"
        return mask, kps

class SimTooltipDataset():
    def __init__(self, mask_path, kps_path, mode, obj_num=1, transform=None):
        self.mask_path = mask_path
        self.kps_file = kps_path
        self.mode = mode
        self.transform = transform
        self.is_inst = False
        self.obj_num = obj_num

        self.mask_list = sorted(os.listdir(mask_path))
        self.frame_list = [mask.split('.')[0] for mask in self.mask_list]

        kps = json.load(open(self.kps_file, "r"))
        self.tooltip_idx = ["l_gripper_keypoint", "r_gripper_keypoint"]
        self.good_kps = []
        self.good_frames = []
        for frame in self.frame_list:
            kps_frame_full = kps[frame]

            ## check if the keypoints are valid, if not, skip
            if kps_frame_full[self.tooltip_idx[0]] is None or kps_frame_full[self.tooltip_idx[1]] is None:
                # print(f"Invalid keypoints at frame {frame} - null")
                continue
            
            if kps_frame_full[self.tooltip_idx[0]][0][0] < 0 \
                or kps_frame_full[self.tooltip_idx[0]][1][0] < 0 \
                    or kps_frame_full[self.tooltip_idx[1]][0][0] < 0 \
                        or kps_frame_full[self.tooltip_idx[1]][1][0] < 0:
                print(f"Invalid keypoints at frame {frame} - negative")
                continue

            kps_frame_tip = []
            for tooltip in self.tooltip_idx:
                    kps_frame_tip.append((kps_frame_full[tooltip][0][0], kps_frame_full[tooltip][1][0]))

            self.good_kps.append(kps_frame_tip)
            self.good_frames.append(frame+".png")
    
    def __len__(self):
        return len(self.good_frames)
    
    def __getitem__(self, idx):
        mask = cv2.imread(os.path.join(self.mask_path, self.good_frames[idx]))
        kps = self.good_kps[idx]

        ## check if the keypoints are valid, if not, skip
        for kp in kps:
            if kp[0] < 0 or kp[1] < 0:
                print(f"Invalid keypoints at frame {self.good_frames[idx]}")
                continue
                # return self.__getitem__(idx+1)

        ## transform
        if self.mode == "train":
            self.transform = TooltipTransform()
            mask, kps = self.transform(mask, kps)
            if len(kps) != 2:
                # print(f"Invisible keypoints at frame {self.good_frames[idx]} after transform")
                return self.__getitem__(idx+1)
        if self.mode == "val":
            self.transform = TooltipInferTransform()
            mask, kps = self.transform(mask, kps)

        ## generate instance-level mask label
        if self.is_inst:
            mask_inst = np.where(np.any(mask > 0, axis=-1), 255, 0).astype(np.uint8)
            mask_inst_vis = cv2.merge((mask_inst, mask_inst, mask_inst))
            mask = mask_inst_vis
        ## draw keypoints on mask
        # for kp in kps:
        #     mask_inst_vis = cv2.circle(mask_inst_vis, (int(kp[0]), int(kp[1])), 5, (255, 0, 0), -1)
        # plt.imshow(mask_inst_vis)
        # plt.show()

        ## convert to tensor
        # if self.is_inst:
        #     mask = cv2.merge((mask_inst, mask_inst, mask_inst))

        # mask = torch.from_numpy(mask).permute(2, 0, 1)
        mask = torch.from_numpy(mask).permute(2, 0, 1)
        kps = torch.tensor(kps)

        assert kps.shape[0] == 2, f"{self.good_frames[idx]}, {kps.shape} keypoints should be in shape (2, n_kps)"
        return mask, kps

class MixedTooltipDataset():
    def __init__(self, simulated_data: torch.utils.data.Dataset, real_data: torch.utils.data.Dataset):
        self.simulated_data = simulated_data
        self.real_data = real_data
        self.simulated_len = len(simulated_data)
        self.real_len = len(real_data)
        self.total_len = self.simulated_len + self.real_len
        print(f"Real dataset size: {self.real_len}")
        print(f"Simulated dataset size: {self.simulated_len}")
        print(f"Mixed dataset size: {self.total_len}")

    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        if idx < self.real_len:
            # Return a sample from the real dataset
            return self.real_data[idx]
        else:
            # Return a sample from the simulated dataset
            return self.simulated_data[idx - self.real_len]

class GreenScreenInferDataset():
    def __init__(self, mask_path):
        self.mask_path = mask_path
        self.is_inst = False

        self.mask_list = sorted(os.listdir(mask_path))
        self.frame_list = [int(mask.split('.')[0][5:]) for mask in self.mask_list]
        
    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, idx):
        mask = cv2.imread(os.path.join(self.mask_path, self.mask_list[idx]))
        mask = cv2.resize(mask, (640, 512))
        # convert to tensor
        mask = torch.from_numpy(mask).permute(2, 0, 1) # (C, H, W)
        return mask


class TooltipTransform():
    def __init__(self):
        super().__init__()
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # A.Rotate(limit=45, p=0.5),
            A.Affine(scale=(0.5, 1.0), p=0.2),
            A.Resize(512, 640), # resize to 512x640
        ], keypoint_params=A.KeypointParams(format='xy'))            

    def __call__(self, mask, kps):
        # albumentations transform
        transformed = self.transform(image=mask, keypoints=kps)
        mask = transformed['image']
        kps = transformed['keypoints']

        return mask, kps
    
class TooltipInferTransform():
    def __init__(self):
        super().__init__()
        self.transform = A.Compose([
            A.Resize(512, 640), # resize to 512x640
        ], keypoint_params=A.KeypointParams(format='xy'))            

    def __call__(self, mask, kps):
        # albumentations transform
        transformed = self.transform(image=mask, keypoints=kps)
        mask = transformed['image']
        kps = transformed['keypoints']

        return mask, kps

if __name__ == "__main__":
    sim_mask_path = "/bigdata/SurgPose/tooltip/simulated_data"
    real_mask_path = "/bigdata/SurgPose/tooltip/train"
    val_mask_path   = "/bigdata/SurgPose/tooltip/eval"

    sim_kps_file  = "/bigdata/SurgPose/tooltip/keypoints.json"
    real_kps_file = "/bigdata/SurgPose/tooltip/keypoints_100002_left.yaml"
    val_kps_file    = "/bigdata/SurgPose/tooltip/keypoints_100003_left.yaml"

    # train_dataset = TooltipDataset(train_mask_path, train_kps_file, mode="train")
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    sim_dataset = SimTooltipDataset(sim_mask_path, sim_kps_file, mode="train")
    real_dataset = TooltipDataset(real_mask_path, real_kps_file, mode="train")
    mixed_dataset = MixedTooltipDataset(sim_dataset, real_dataset)
    val_dataset = TooltipDataset(val_mask_path, val_kps_file, mode="val")

    train_dataloader  = torch.utils.data.DataLoader(mixed_dataset, batch_size=8, shuffle=True, num_workers=16)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=16)

    ## visualize the entire training set
    # os.makedirs("training_set_vis", exist_ok=True)

    # for idx, (mask, kps) in enumerate(train_loader):
    #     mask = mask[0].permute(1, 2, 0).numpy().astype(np.uint8)
    #     mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    #     for kp in kps[0]:
    #         mask = cv2.circle(mask, (int(kp[0]), int(kp[1])), 5, (255, 0, 0), -1)
    #     plt.imshow(mask)
    #     plt.axis('off')
    #     plt.savefig(f"training_set_vis/{idx}.png", bbox_inches="tight", pad_inches=0)
    #     # plt.show()
    #     # breakpoint()

    ## check the first batch of the training set
    # dataiter = iter(train_dataloader) # sanity check
    # images, labels = next(dataiter)
    # imshow(torchvision.utils.make_grid(images))

    # breakpoint()

    ## check the first batch of the validation set
    dataiter = iter(val_loader) # sanity check
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images))

    # val_dataset = TooltipDataset(val_mask_path, val_kps_file, mode="val")
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=16)
    # dataiter = iter(val_loader) # sanity check
    # images, labels = next(dataiter)
    # imshow(torchvision.utils.make_grid(images))