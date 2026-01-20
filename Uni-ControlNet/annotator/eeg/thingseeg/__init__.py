
import torch.nn.functional as F
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

from .ATMS import ATMS
from .ATMS_crossatt import ATMS_crossatt
from .LSTM import EEGFeatNetMultiHead
import torch
import torch.nn as nn

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
        self.eeg_encoder = nn.Module()
        self.eeg_encoder.module = eeg_encoder
        self.contra_head_eeg = Contra_head(eeg_dim,1024)

    def forward(self, x, subject_ids = None, encoder_type=None):
        #encode
        #forward call
        features = self.eeg_encoder.module(x)
            
        # projection
        out = self.contra_head_eeg(features)
        
        #normalizeq
        out = F.normalize(out,dim=-1)
        return out
    

class EEGDetectorTHINGS:
    def __init__(self, encoder_type, ckpt_path):
        if encoder_type == "THINGS_ATMS":
            eeg_encoder = ATMS()
            ckpt = torch.load(ckpt_path, map_location='cpu')
            state_dict = {}
            for k, v in ckpt.items():
                new_key = k.replace("eeg_encoder.", "")  # adjust to match the model’s hierarchy
                state_dict[new_key] = v
        elif encoder_type == "THINGS_LSTM_crossatt":
            eeg_encoder = EEGFeatNetMultiHead(in_channels=250, projection_dim=512, num_layers=4)
            eeg_encoder = EEGEncoderWithProjection(eeg_encoder, eeg_dim=512, encoder_type=encoder_type)
            ckpt = torch.load(ckpt_path, map_location='cpu')
            state_dict = {}
            for k, v in ckpt.items():
                state_dict[k] = v
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        
        
        missing, unexpected = eeg_encoder.load_state_dict(state_dict,strict=False )
        print(f"{encoder_type} Encoder loaded. (hoepfully correctly): \nMissing keys:", missing)#, "\nUnexpected keys:", unexpected)
        self.model = eeg_encoder.cuda().eval()

    def __call__(self, eeg, subject_id):
        #assert img.ndim == 3
        #logger.info("CALLED")
        with torch.no_grad():
            #logger.info(eeg.shape) #here its a numpy array
            eeg_tensor = torch.as_tensor(eeg).unsqueeze(0) #batch dim
            emb = self.model(eeg_tensor.to('cuda'), subject_id.to('cuda'))
            #logger.info(emb.shape)
            emb = F.normalize(emb, dim=-1)
            feature = emb
            #logger.info(feature.shape)
            
        return feature