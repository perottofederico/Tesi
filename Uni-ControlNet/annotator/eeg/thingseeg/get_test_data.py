# Load the dataset
from datasets import load_dataset
import numpy as np
import hashlib
from PIL import Image
import torch
import sys
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import os
if './' not in sys.path:
	sys.path.append('./')
     

dataset = load_dataset(
    "perottofederico/things-eeg-captions-single-sub_08",
    split="test_sub_08"
)
print("test Dataset loaded")

os.makedirs("./data_THINGSEEG/conditions/eeg/validation/", exist_ok=True)
os.makedirs("./data_THINGSEEG/images/validation/", exist_ok=True)
anno_path = "./data_THINGSEEG/anno_test.txt"

if not os.path.exists(anno_path):
    open(anno_path, "w").close()

with open(anno_path, "w") as f:
    for i,sample in enumerate(dataset):
        file_id = f"{sample['img_id']}_{int(sample['subject'])}"
        caption = sample['caption']
        eeg_data = torch.tensor(sample['eeg'])
        image = sample['image']
        np.save(f"./data_THINGSEEG/conditions/eeg/validation/{file_id}.npy", eeg_data.squeeze(0).detach().cpu().numpy().astype(np.float32), allow_pickle=False)

        # Save image as .jpg file
        img = image.save(f"./data_THINGSEEG/images/validation/{file_id}.jpg")
        f.write(f"{file_id}\t{caption}\n")
        if i % 100 == 0: print(f"Processed {i} samples.", end="\r")
print("\nDone.")