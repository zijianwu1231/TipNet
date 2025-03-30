import os
import json

# mask_list = sorted(os.listdir("/bigdata/SurgPose/tooltip/new_train"))
# kps_file = "/bigdata/SurgPose/tooltip/new_kps_train.json"

# with open(kps_file, "r") as f:
#     kps = json.load(f)

# for mask_name in mask_list:
#     new_name = f"{str(int(mask_name.split('.')[0])+12097).zfill(6)}.png"
#     os.rename(f"/bigdata/SurgPose/tooltip/new_train/{mask_name}", f"/bigdata/SurgPose/tooltip/new_train_rename/{new_name}")

#     kps[str(int(mask_name.split('.')[0])+12097).zfill(6)] = kps.pop(mask_name.split('.')[0])

# # new json file "new_keypoints.json"
# with open("kps_train_rename.json", "w") as f:
#     json.dump(kps, f)

sim_kps_file = "/bigdata/SurgPose/tooltip/keypoints.json"
real_kps_file = "/bigdata/SurgPose/tooltip/new_kps_train_rename.json"

with open(sim_kps_file, "r") as f:
    sim_kps = json.load(f)
print(len(sim_kps))

breakpoint()

with open(real_kps_file, "r") as f:
    real_kps = json.load(f)
print(len(real_kps))

# combine the two json files
breakpoint()
sim_kps.update(real_kps)


print(len(sim_kps))

breakpoint()

# new json file "new_keypoints.json"
with open("keypoints_mixed.json", "w") as f:
    json.dump(sim_kps, f)