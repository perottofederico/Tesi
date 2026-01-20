# Load the dataset
from datasets import load_dataset
import numpy as np
import hashlib
from PIL import Image
import torch
import sys
from ATMS import ATMS
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import os
if './' not in sys.path:
	sys.path.append('./')
     

dataset = load_dataset(
    "perottofederico/things-eeg-captions",
    split="test_sub_08"
)
print("test Dataset loaded")

os.makedirs("./data_THINGSEEG/text_features/", exist_ok=True)
os.makedirs("./data_THINGSEEG/images/validation/", exist_ok=True)

for i,sample in enumerate(dataset):
    file_id = f"{sample['img_id']}_{int(sample['subject'])}"
    text_feature = torch.tensor(sample['text_features'])
    np.save(f"./data_THINGSEEG/text_features/{file_id}.npy", text_feature.numpy().astype(np.float32), allow_pickle=False)
    if i % 100 == 0: print(f"Processed {i} samples.")