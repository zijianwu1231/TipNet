import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

mask_list = sorted(os.listdir("/bigdata/SurgPose/tooltip/l_mask"), key=lambda x: int(x.split(".")[0][5:]))

for idx, mask_file in enumerate(mask_list):
    mask = cv2.imread(f"/bigdata/SurgPose/tooltip/l_mask/{mask_file}", cv2.IMREAD_GRAYSCALE)
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) # 0, 10, 20, 30
    # mask = cv2.resize(mask, (1400, 986))

    # convert the single-channel mask to 3-channel image, 10 to [255, 0, 0], 20 to [0, 255, 0], 30 to [0, 0, 255]
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    # Create an empty 3-channel image
    height, width = mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    color_map = {
        0:   (0, 0, 0),       # Black
        10:  (255, 0, 0),     # Blue
        20:  (0, 255, 0),     # Green
        30:  (0, 0, 255),     # Red
    }

    # Apply color mapping
    for val, color in color_map.items():
        color_mask[mask == val] = color

    color_mask = cv2.resize(color_mask, (1400, 986))

    # plt.imshow(color_mask)
    # plt.show()
    # breakpoint()

    cv2.imwrite(f"/bigdata/SurgPose/tooltip/l_mask/{mask_file}", color_mask)
