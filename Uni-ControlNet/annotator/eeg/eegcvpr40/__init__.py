import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from .LSTM import EEGFeatNetMultiHead, EEGFeatNet
from .controlnet_conditioning_eeg import create_model
from annotator.util import annotator_ckpts_path

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Remove old handlers if any
for h in list(logger.handlers):
    logger.removeHandler(h)

fh = logging.FileHandler("gradio_debug.log", mode="w", encoding="utf-8")
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

class Contra_head(nn.Module):
    def __init__(self, input_dim, contra_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, contra_dim, bias=False)
    def forward(self, cls_token):
        return self.linear(cls_token)

class EEGEncoderWithProjection(nn.Module):
    def __init__(self, eeg_encoder, eeg_dim, encoder_type):
        super(EEGEncoderWithProjection, self).__init__()
        self.encoder_type = encoder_type
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
        if self.encoder_type == "LSTM" or self.encoder_type == "LSTM_crossatt":
            features = self.eeg_encoder.module(x)
        else: #GWIT
            features = self.eeg_encoder(x, subject_ids)
        
        #pooling 
        if self.encoder_type == "GWIT":
            features =features.mean(dim=[2,3])
            
        # projection
        out = self.contra_head_eeg(features)
        
        #normalize
        out = F.normalize(out,dim=-1)
        return out



class EEGDetector:
    def __init__(self, encoder_type, ckpt_path):
        self.encoder_type = encoder_type
        if encoder_type == "LSTM":
            eeg_encoder = EEGFeatNet(n_features=128, projection_dim=1024, num_layers=4).to("cuda")
            eeg_dim = 128
        elif encoder_type == "LSTM_crossatt":
            eeg_encoder = EEGFeatNetMultiHead(projection_dim=512, num_layers=4)
            eeg_dim = 512
        else: #GWIT
            eeg_encoder = create_model()
            eeg_dim = 320
            
        self.model = EEGEncoderWithProjection(eeg_encoder, eeg_dim, encoder_type).to('cuda')
        missing, unexpected = self.model.load_state_dict(torch.load(ckpt_path, map_location='cpu'),strict=False )
        print(f"{encoder_type} Encoder loaded. (hoepfully correctly): \nMissing keys: {missing}")#, "\nUnexpected keys:", unexpected)
        self.model = self.model.cuda().eval()

    def __call__(self, eeg, subject_id=None):
        #assert img.ndim == 3
        #logger.info("CALLED")
        with torch.no_grad():
            #logger.info(eeg.shape) #here its a numpy array
            eeg_tensor = torch.as_tensor(eeg).unsqueeze(0) #batch dim
            eeg_view = eeg_tensor.view(-1,eeg_tensor.shape[2],eeg_tensor.shape[1]).to('cuda')
            #logger.info(eeg_view.shape)
            if(self.encoder_type == "LSTM" or self.encoder_type == "LSTM_crossatt"):
                emb = self.model(eeg_view)
            else:
                emb = self.model(eeg_view, subject_id.to('cuda'))
            #logger.info(emb.shape)
            feature = emb.squeeze(0).detach().cpu().numpy()
            #logger.info(feature.shape)
        return feature