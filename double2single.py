import os
import cv2
import numpy as np
import yaml
import json
from matplotlib import pyplot as plt

mask_path = "/bigdata/SurgPose/tooltip/train"
mask_list = sorted(os.listdir(mask_path), key=lambda x: int(x.split(".")[0][5:]))

kps_path = "/bigdata/SurgPose/tooltip/keypoints_100002_left.yaml"
with open(kps_path, "r") as f:
    kps = yaml.load(f, Loader=yaml.FullLoader)

tooltip_idx = [4, 5, 11, 12]
new_kps = {}

for idx, mask_name in enumerate(mask_list):
    mask_id = int(mask_name.split(".")[0][5:])
    kps_frame = kps[mask_id]

    kps_frame_tip = []
    for tip_idx in tooltip_idx:
        kps_frame_tip.append(tuple(kps_frame[tip_idx]))

    mask = cv2.imread(os.path.join(mask_path, mask_name))
    mask_inst = np.where(np.any(mask > 0, axis=-1), 255, 0).astype(np.uint8)

    # Only keep one object
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_inst, connectivity=8)
    inst_idx = np.random.randint(1, n_labels)
    mask_inst_single = np.where(labels == inst_idx, 255, 0).astype(np.uint8)
    mask_inst_bool = np.where(labels == inst_idx, True, False)
    mask_vis = np.where(labels == inst_idx, 1, 0).astype(np.uint8)
    mask_vis = cv2.merge((mask_vis, mask_vis, mask_vis))

    # keep the keypoints within the mask
    kps_frame_tip = [kp for kp in kps_frame_tip if mask_inst_bool[int(kp[1]), int(kp[0])]]

    # image is scaled from 1400x986 to 640x512, keypoints are also scaled
    kps_frame_tip = [(kp[0] * 640 / 1400, kp[1] * 512 / 986) for kp in kps_frame_tip]

    if len(kps_frame_tip) != 2:
        print(f"Frame {mask_id} has {len(kps_frame_tip)} keypoints")
        continue

    assert len(kps_frame_tip) == 2, "There should be 2 keypoints"

    mask = mask * mask_vis

    os.makedirs("new_train", exist_ok=True)

    print(idx)

    mask = cv2.resize(mask, (640, 512))
    cv2.imwrite(f"new_train/{str(idx).zfill(6)}.png", mask)

    for kp in kps_frame_tip:
        mask = cv2.circle(mask, (int(kp[0]), int(kp[1])), 5, (255, 255, 255), -1)

    mask_name = "check_" + str(idx).zfill(6) + ".png"
    cv2.imwrite(f"new_train/{mask_name}", mask)

    new_kps[str(idx).zfill(6)] = {"l_gripper_keypoint" : [[kps_frame_tip[0][0]],[kps_frame_tip[0][1]]],
                                  "r_gripper_keypoint" : [[kps_frame_tip[1][0]],[kps_frame_tip[1][1]]]}

    # plt.imshow(mask)
    # plt.show()

print(len(new_kps))

# save the new keypoints as .json
with open("new_kps_train.json", "w") as f:
    json.dump(new_kps, f)