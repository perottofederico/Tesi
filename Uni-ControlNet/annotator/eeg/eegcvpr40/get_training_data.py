# Load the dataset
from datasets import load_dataset
import numpy as np
import hashlib
from PIL import Image
import torch
import sys
from LSTM import EEGFeatNetMultiHead, EEGFeatNet
from controlnet_conditioning_eeg import ControlNetEEGConditioningEmbedding
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import os 

if './' not in sys.path:
	sys.path.append('./')
     

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
    


def load_encoder(encoder_type, ckpt_path):
    if encoder_type == "LSTM":
        eeg_encoder = EEGFeatNet(n_features=128, projection_dim=1024, num_layers=4).to("cuda")
        eeg_dim = 128
        #ckpt_path = "./annotator/ckpts/subj4_only_best_ret_area_LSTM.pt"
        
    elif encoder_type == "LSTM_crossatt":
        #ckpt_path = "./annotator/ckpts/LSTM_crossatt_bestarea.pt"
        eeg_encoder = EEGFeatNetMultiHead(projection_dim=512, num_layers=4)
        eeg_dim = 512
        
    elif encoder_type == "GWIT":
        from controlnet_conditioning_eeg import create_model
        eeg_dim = 320
        eeg_encoder = create_model()
        #ckpt_path = "./annotator/ckpts/GWIT_fixedtemp_nosubjlayers.pt"
    else:
        raise ValueError(f"Not implemented: {encoder_type}")

    model = EEGEncoderWithProjection(eeg_encoder, eeg_dim, encoder_type)
    #use model weights from ckpts folder
    ckpt = torch.load(ckpt_path, map_location='cpu')
    missing, unexpected = model.load_state_dict(ckpt,strict=False)
    model = model.to('cuda')
    model.requires_grad_(False)
    model.eval()
    print("EEG Encoder loaded. (hoepfully correctly): \nMissing keys:", missing)#, "\nUnexpected keys:", unexpected)
    return model

def load_ds():
    dataset = load_dataset(
        "perottofederico/EEGCVPR40-captions",
        split="train"
    )
    #only for single subject training
    #dataset = dataset.filter(lambda x: x['subject'] == 4)

    print("Dataset loaded and preprocessed.")
    return dataset
    
    
def generate_embeddings(encoder_type, model, dataset):
    
    os.makedirs("./data_EEGCVPR/conditions/eeg", exist_ok=True)
    os.makedirs("./data_EEGCVPR/images", exist_ok=True)
    anno_path = "./data_EEGCVPR/anno.txt"
    if not os.path.exists(anno_path):
        open(anno_path, "w").close()
    
    #Iterate thorugh the dataset, for each sample:
    # Pass the eeg data through the eeg encoder to get the conditioning vector
    # Save the conditioning vector as a <id>.npy file in the appropriate directory
    # Save the image as a <id>.jpg file in the appropriate directory
    # add a line to anno.txt with the file correct id and caption
    with open(anno_path, "w") as f:
        for i,sample in enumerate(dataset):
            # since there are duplicate images, using only the image hash would overwrite eeg signals
            # (they would sahre the same name but are actually different)
            # So we add the subject number to the file id to make it unique
            file_id = f"{sample['img_id']}_{int(sample['subject'])}"
            caption = sample['caption']
            eeg_data = torch.tensor(sample['conditioning_image'])
            image = sample['image']
            # Process eeg_data through eeg encoder to get conditioning vector
            eeg_tensor = torch.as_tensor(eeg_data).unsqueeze(0) #batch dim
            eeg_view = eeg_tensor.view(-1,eeg_tensor.shape[2],eeg_tensor.shape[1]).to('cuda')
            with torch.no_grad():
                cond = model(eeg_view, torch.tensor(int(sample['subject'])).unsqueeze(0).cuda(), encoder_type)
            # Save conditioning vector as .npy file
            np.save(f"./data_EEGCVPR/conditions/eeg/{file_id}.npy", cond.squeeze(0).detach().cpu().numpy().astype(np.float32), allow_pickle=False)
            # Save image as .jpg file
            img = image.save(f"./data_EEGCVPR/images/{file_id}.jpg")
            # Append to anno.txt
            f.write(f"{file_id}\t{caption}\n")
            if i % 1000 == 0: print(f"Processed {i} samples.")
        
        
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=["LSTM", "LSTM_crossatt", "GWIT"], required=True,
                        help=" EEG encoder to use to precompute the eeg embeddings.")
    parser.add_argument("--ckpt_path", type=str, required=True)
    args = parser.parse_args()

    model = load_encoder(args.encoder, args.ckpt_path)
    dataset = load_ds()
    generate_embeddings(args.encoder, model, dataset)

if __name__ == "__main__":
    main()