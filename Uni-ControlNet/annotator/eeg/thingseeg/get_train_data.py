# Load the dataset
from datasets import load_dataset
import numpy as np
import hashlib
from PIL import Image
import torch
import sys
import os
if './' not in sys.path:
	sys.path.append('./')
from annotator.eeg.thingseeg.ATMS import ATMS
from annotator.eeg.thingseeg.LSTM import EEGFeatNetMultiHead
#from ATMS_crossatt import ATMS_crossatt
import tqdm
import torch.nn as nn
import torch.nn.functional as F


class Contra_head(nn.Module):
    def __init__(self, input_dim, contra_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, contra_dim, bias=False)
    def forward(self, cls_token):
        return self.linear(cls_token)

class EEGEncoderWithProjection(nn.Module):
    def __init__(self, eeg_encoder, eeg_dim, encoder_type):
        super(EEGEncoderWithProjection, self).__init__()
        if encoder_type == "LSTM" or encoder_type == "LSTM_crossatt":
            self.eeg_encoder = nn.Module()
            self.eeg_encoder.module = eeg_encoder
        else: #GWIT
            self.eeg_encoder = eeg_encoder
        self.contra_head_eeg = Contra_head(eeg_dim,1024)

    def forward(self, x, subject_ids = None, encoder_type=None):
        #encode
        #features = self.eeg_encoder.module(x, subject_ids if subject_ids is not None else None)

        #forward calls
        if encoder_type == "LSTM" or encoder_type == "LSTM_crossatt":
            features = self.eeg_encoder.module(x)
        else:
            features = self.eeg_encoder(x, subject_ids)
        
        #pooling depending on the encoder type
        if encoder_type == "GWIT":
            features =features.mean(dim=[2,3])
            
        # projection
        out = self.contra_head_eeg(features)
        
        #normalize
        out = F.normalize(out,dim=-1)
        return out

def load_ds():
    dataset = load_dataset(
        "perottofederico/things-eeg-captions-single-sub_08",
        split="train_sub_08"
    )

    print("Dataset loaded.")
    return dataset

def load_encoder(encoder_type, ckpt_path):
    if encoder_type == "ATMS":
        model = ATMS()
        
    elif encoder_type == "LSTM_crossatt":
        #ckpt_path = "./annotator/ckpts/LSTM_crossatt_bestarea.pt"
        model = EEGFeatNetMultiHead(in_channels=250, projection_dim=512, num_layers=4)
        eeg_dim = 512
        model = EEGEncoderWithProjection(model, eeg_dim, encoder_type)
        
    else:
        raise ValueError(f"Not implemented: {encoder_type}")

    #use model weights from ckpts folder
    ckpt = torch.load(ckpt_path, map_location='cpu')
    missing, unexpected = model.load_state_dict(ckpt,strict=False)
    model = model.to('cuda')
    model.requires_grad_(False)
    model.eval()
    print(f"{encoder_type} EEG Encoder loaded: \nMissing keys:", missing)#, "\nUnexpected keys:", unexpected)
    return model

'''
#Load the eeg encoder (possibly whole gram model)
eeg_encoder = ATMS().to('cuda')

#use model weights from ckpts folder
print("Current dir:", os.getcwd())
#ckpt_path = "./annotator/ckpts/ATMS_sub08_best_ret_area.pt" #no cross attention
ckpt_path = "./annotator/ckpts/ATMS_sub08_best_ret_area.pt"
ckpt = torch.load(ckpt_path, map_location='cpu')
state_dict = {}
for k, v in ckpt.items():
    new_key = k.replace("eeg_encoder.", "")  # adjust to match the model’s hierarchy
    state_dict[new_key] = v

missing, unexpected = eeg_encoder.load_state_dict(state_dict, strict=False)
eeg_encoder.eval()
print("EEG Encoder loaded. (hoepfully correctly): \nMissing keys:", missing)#, "\nUnexpected keys:", unexpected)
'''


def generate_embeddings(encoder_type, eeg_encoder, dataset):
    #Iterate thorugh the dataset, for each sample:
    # Pass the eeg data through the eeg encoder to get the conditioning vector
    # Save the conditioning vector as a <id>.npy file in the appropriate directory
    # Save the image as a <id>.jpg file in the appropriate directory
    # add a line to anno.txt with the file correct id and caption
    
    os.makedirs("./data_THINGSEEG/conditions/eeg", exist_ok=True)
    os.makedirs("./data_THINGSEEG/images", exist_ok=True)
    anno_path = "./data_THINGSEEG/anno.txt"
    if not os.path.exists(anno_path):
        open(anno_path, "w").close()

    with open(anno_path, "w") as f:
        for i,sample in enumerate(dataset):
            # since there are duplicate images, using only the image hash would overwrite eeg signals
            # (they would sahre the same name but are actually different)
            # So we add the subject number to the file id to make it unique
            file_id = f"{sample['img_id']}_{int(sample['subject'])}_{i}"
            caption = sample['caption']
            eeg_data = torch.tensor(sample['eeg'])
            image = sample['image']
            # Process eeg_data through eeg encoder to get conditioning vector
            eeg_tensor = torch.as_tensor(eeg_data).unsqueeze(0) #batch dim
            subject_id = torch.as_tensor(int(sample['subject'])).unsqueeze(0)
            with torch.no_grad():
                if encoder_type == "ATMS":
                    cond = eeg_encoder(eeg_tensor.cuda(), subject_id.cuda())
                    cond = F.normalize(cond,dim=-1) 
                elif encoder_type == "LSTM_crossatt":
                    cond = eeg_encoder(eeg_tensor.cuda(), encoder_type=encoder_type)
                    #projection and normalization done inside the model
            
            # Save conditioning vector as .npy file
            #remember to switch folder names if already ran
            #TODO add creatin|g folders if not exist
            np.save(f"./data_THINGSEEG/conditions/eeg/{file_id}.npy", cond.squeeze(0).detach().cpu().numpy().astype(np.float32), allow_pickle=False)
            # Save image as .jpg file
            img = image.save(f"./data_THINGSEEG/images/{file_id}.jpg")
            # Append to anno.txt
            f.write(f"{file_id}\t{caption}\n")
            if i % 1000 == 0: print(f"Processed {i} samples.", end='\r')
    print("\nDone!")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=["ATMS", "LSTM_crossatt"], required=True,
                        help=" EEG encoder to use to precompute the eeg embeddings.")
    parser.add_argument("--ckpt_path", type=str, required=True)
    args = parser.parse_args()

    model = load_encoder(args.encoder, args.ckpt_path)
    dataset = load_ds()
    generate_embeddings(args.encoder, model, dataset)

if __name__ == "__main__":
    main()