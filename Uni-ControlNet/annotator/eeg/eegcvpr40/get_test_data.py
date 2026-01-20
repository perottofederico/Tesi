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
if './' not in sys.path:
	sys.path.append('./')
import os


dataset = load_dataset(
    "perottofederico/EEGCVPR40-captions",
    split="test"
)

os.makedirs("./data_EEGCVPR/conditions/eeg/validation/", exist_ok=True)
os.makedirs("./data_EEGCVPR/images/validation/", exist_ok=True)
anno_path = "./data_EEGCVPR/test_labels.txt"
if not os.path.exists(anno_path):
    open(anno_path, "w").close()
    

#dataset = dataset.map(preprocess, batched=True)
print("Dataset loaded and preprocessed.")
with open(anno_path, "w") as f:
    for i,sample in enumerate(dataset):
        file_id = f"{sample['img_id']}_{int(sample['subject'])}"
        caption = sample['label']
        #eeg_data = torch.tensor(sample['conditioning_image'])
        #image = sample['image']
        #np.save(f"./data_EEGCVPR/conditions/eeg/validation/{file_id}.npy", eeg_data.squeeze(0).detach().cpu().numpy().astype(np.float32), allow_pickle=False)

        # Save image as .jpg file
        #img = image.save(f"./data_EEGCVPR/images/validation/{file_id}.jpg")
        f.write(f"{file_id}\t{caption}\n")
        if i % 1000 == 0: print(f"Processed {i} samples.", end="\r")