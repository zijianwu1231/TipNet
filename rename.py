import os

mask_list = sorted(os.listdir("/bigdata/SurgPose/tooltip/eval"), key=lambda x: int(x.split(".")[0][5:]))

for idx, mask in enumerate(mask_list):
    new_name = f"{str(idx).zfill(6)}.png"
    os.rename(f"/bigdata/SurgPose/tooltip/eval/{mask}", f"/bigdata/SurgPose/tooltip/eval/{new_name}")