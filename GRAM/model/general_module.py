import math
import os
import sys
import torch
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from easydict import EasyDict as edict
from torch.nn import LayerNorm as LayerNorm
from utils.logger import LOGGER

from transformers import AutoTokenizer, PretrainedConfig
from model.eeg_encoders.ATMS import ATMS
from model.eeg_encoders.ATMS_crossatt import ATMS_crossatt
from model.eeg_encoders.LSTM import EEGFeatNet, EEGFeatNetMultiHead

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel

    else:
        raise ValueError(f"{model_class} is not supported.")

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    
class GELU(nn.Module):
    def forward(self, input_):
        output = gelu(input_)
        return output


class Contra_head(nn.Module):
    def __init__(self, input_dim, contra_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, contra_dim, bias=False)
    def forward(self, cls_token):
        return self.linear(cls_token)


class Match_head(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.activation = GELU()
        self.layernorm = LayerNorm(hidden_size, eps=1e-12)
        self.linear2 = nn.Linear(hidden_size, 2)
    def forward(self, cls_token):
        return self.linear2(self.layernorm(self.activation(self.linear1(cls_token))))


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class TokenMasker(nn.Module):
    def __init__(self, mask_token = -1, range_start=-1, range_end=-1):
        super().__init__()
        self.mask_token = mask_token
        self.range = [range_start,range_end]

    def forward(self, tokens, mask_prob):
        tokens = tokens.clone() ### important, must have
        tokens, labels = self.perform_mask(tokens, mask_prob)
        return tokens, labels

    
    def perform_mask(self, tokens, mask_prob):
        
        tokens = np.array(tokens.cpu().numpy())

        ### generate indicator first:
        mask_indicator = np.zeros(tokens.shape, dtype=np.int64)
        for i in range(len(mask_indicator)):
            while all(mask_indicator[i] == 0):
                for j in range(1, len(mask_indicator[0])):
                    if tokens[i][j]!=0 and random.random() < mask_prob:
                        mask_indicator[i][j] = 1
        
        


        labels = -np.ones(tokens.shape, dtype=np.int64) * 100 ### -100 ignore idx for nn.CrossEntropyLoss used in BERT
        for i in range(tokens.shape[0]):
            for j in range(tokens.shape[1]):
                
                if mask_indicator[i][j] == 1 :
                    src_token = tokens[i][j]
                    prob = random.random()   #### e-6 too much time
                    if prob < 0.8:
                        tokens[i][j] = self.mask_token  ### e-6 have no idea why too much 
                    elif prob < 0.9: 
                        tokens[i][j] = random.choice(list(range(*self.range)))   
                    #tokens[i][j] = self.mask_token
                    labels[i][j] = src_token


        tokens =torch.from_numpy(tokens).long().cuda()
        labels =torch.from_numpy(labels).long().cuda()
        
        return tokens, labels


class MMGeneralModule(nn.Module):
    def __init__(self):
        super().__init__()
      
    
    def modify_checkpoint(self, checkpoint):
        new_ckpt = {}
        for k,v in checkpoint.items():
            if 'video' in k:
                new_ckpt[k.replace('video','vision')]=v
            elif 'evaclip_model' in k:
                new_ckpt[k.replace('evaclip_model','vision_encoder')]=v
            elif 'clip_model' in k:    
                new_ckpt[k.replace('clip_model','vision_encoder')]=v
            else:
                new_ckpt[k] = v.float()
        
        checkpoint = new_ckpt

    
        # if self.config.vision_resolution != pretrain_cfg['vision_resolution']:
        if self.config.vision_encoder_type.startswith('clip'):
            vision_width = checkpoint["vision_encoder.visual.positional_embedding"].shape[1]
            vision_layers = len([k for k in checkpoint.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = checkpoint["vision_encoder.visual.conv1.weight"].shape[-1]
            
            grid_size = round((checkpoint["vision_encoder.visual.positional_embedding"].shape[0] - 1) ** 0.5)
       
            src  = checkpoint["vision_encoder.visual.positional_embedding"]
            src_cls = src[0:1]
            src_oth = src[1:]
            new_grid_size = self.config.vision_resolution // vision_patch_size
            if new_grid_size!=grid_size:
                src_oth = F.interpolate(src_oth.reshape(grid_size,grid_size,vision_width).permute(2,0,1).unsqueeze(0),(new_grid_size,new_grid_size),mode='bilinear')
                src_oth = src_oth[0].permute(1,2,0).reshape(-1,src.shape[-1])
                tgt = torch.cat((src_cls,src_oth),dim=0)
                checkpoint["vision_encoder.visual.positional_embedding"] = tgt

        elif self.config.vision_encoder_type.startswith('evaclip'):

            vision_width = checkpoint["vision_encoder.visual.pos_embed"].shape[2]
            vision_layers = len([k for k in checkpoint.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])

            vision_patch_size = checkpoint["vision_encoder.visual.patch_embed.proj.weight"].shape[-1]
            
            grid_size = round((checkpoint["vision_encoder.visual.pos_embed"].shape[1] - 1) ** 0.5)
     
            src  = checkpoint["vision_encoder.visual.pos_embed"][0]
            src_cls = src[0:1]
            src_oth = src[1:]
            new_grid_size = self.config.vision_resolution // vision_patch_size
            if new_grid_size!=grid_size:
                src_oth = F.interpolate(src_oth.reshape(grid_size,grid_size,vision_width).permute(2,0,1).unsqueeze(0),(new_grid_size,new_grid_size),mode='bilinear')
                src_oth = src_oth[0].permute(1,2,0).reshape(-1,src.shape[-1])
                tgt = torch.cat((src_cls,src_oth),dim=0)
                checkpoint["vision_encoder.visual.pos_embed"] = tgt.unsqueeze(0)



        else:
            pass
        

     
        return checkpoint



    def construct_vision_encoder(self):
        ##### construct vision encoder 
        if self.config.vision_encoder_type == 'none':
            self.vision_dim = 1024
            return
        if self.config.vision_encoder_type.startswith('clip') or self.config.vision_encoder_type.startswith('evaclip'):
            self.load_clip_model() 
        else:
            self.load_clip_model()
    
        if self.config.frozen_vision:
            for k,v in self.vision_encoder.named_parameters():
                v.requires_grad = False 

            self.vision_encoder = self.vision_encoder.eval()
            self.vision_encoder.train = disabled_train


    def construct_eeg_encoder(self):
        if self.config.eeg_encoder_type == 'gwitvec':
            from .eeg_encoders.controlnet_conditioning_eeg import create_model
            self.eeg_encoder = create_model()#ControlNetConditioningEEG(self.config.eeg_dim, self.config.n_subjects, self.config.checkpointing)
            self.eeg_dim = 2560 #vec version. 
        elif self.config.eeg_encoder_type == 'gwit':
            from .eeg_encoders.controlnet_conditioning_eeg import create_model
            self.eeg_encoder = create_model()#ControlNetConditioningEEG(self.config.eeg_dim, self.config.n_subjects, self.config.checkpointing)
            self.eeg_dim = 320 #normal version
        elif self.config.eeg_encoder_type == 'lstm':
            model = EEGFeatNet(n_features=128, projection_dim=self.multimodal_dim, num_layers=4).to("cuda") #projection dim is 128 for pretrained weights
            model = torch.nn.DataParallel(model).to("cuda")
            self.eeg_encoder = model
            self.eeg_dim = 128
        elif self.config.eeg_encoder_type == 'lstm_cross_attention':
            model = EEGFeatNetMultiHead(in_channels= 128 if self.config.dataset == "EEGCVPR" else 63, projection_dim=512, num_layers=4).to("cuda")
            model = torch.nn.DataParallel(model).to("cuda")
            self.eeg_encoder = model
            self.eeg_dim = 512
        elif self.config.eeg_encoder_type == 'lstm_pretrained':
            current_file_path = os.path.abspath(__file__)
            current_dir = os.path.dirname(current_file_path)
            base_dir = os.path.dirname(current_dir)
            ckpt_path = base_dir+"/EEGStyleGAN_ADA/EEG2Feat/Triplet_LSTM/CVPR40/EXPERIMENT_29/bestckpt/eegfeat_all_0.9665178571428571.pth"
            self.eeg_encoder = EEGFeatNet(n_features=128, projection_dim=128, num_layers=4).to("cuda")
            self.eeg_encoder = torch.nn.DataParallel(self.eeg_encoder).to("cuda")
            self.eeg_dim = 128 
            missing, unexpected = self.eeg_encoder.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
            print(f"Loaded EEG encoder weights. Missing: {missing} Unexpected: {unexpected}")

        elif self.config.eeg_encoder_type == 'ATMS' or self.config.eeg_encoder_type == 'ATMS_SDenc':
            self.eeg_encoder = ATMS().to("cuda")
            self.eeg_dim = 1024
            
        elif self.config.eeg_encoder_type == 'ATMS_crossatt':
            from .eeg_encoders.ATMS_crossatt import ATMS_crossatt
            self.eeg_encoder = ATMS_crossatt().to("cuda")
            self.eeg_dim = 250
        else:
            raise NotImplementedError
        
        self.eeg_encoder.requires_grad_(True)

    def load_clip_text_encoder(self):
        #no vision encoder means no text encoder as well
        if self.config.eeg_encoder_type == 'ATMS': 
            self.multimodal_dim = 1024
            return
        self.multimodal_dim = 1024
        '''
        # SD 2.1 text encoder and tokenizer
        print("Loading Tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
                'Manojb/stable-diffusion-2-1-base',
                subfolder="tokenizer",
                revision=None,
                use_fast=False,
            )

        text_encoder_cls = import_model_class_from_model_name_or_path('Manojb/stable-diffusion-2-1-base', revision=None)
        print("Loading Text Encoder")
        self.text_encoder = text_encoder_cls.from_pretrained(
            'Manojb/stable-diffusion-2-1-base', subfolder="text_encoder", revision=None
        )
        #print(self.text_encoder)
        self.multimodal_dim = 1024 #text dim
        self.text_encoder.requires_grad_(False)
        print("Done!")
        '''

    def load_clip_model(self):

        if self.config.vision_encoder_type.startswith('clip'):
            ### openai clip
            from .vision_encoders.clip.clip import build_model
            from .vision_encoders.clip.clip import Transformer
            if  self.config.vision_encoder_type == 'clip_vit_base_16':
                clip_weight = torch.jit.load('./pretrained_weights/clip/ViT-B-16.pt', map_location='cpu')
                self.vision_dim = 768
            elif self.config.vision_encoder_type == 'clip_vit_large_14_336px':
                clip_weight = torch.jit.load('./pretrained_weights/clip/ViT-L-14-336px.pt', map_location='cpu')
                self.vision_dim = 1024
            elif self.config.vision_encoder_type == 'clip_vit_base_32':
                clip_weight = torch.jit.load('./pretrained_weights/clip/ViT-B-32.pt', map_location='cpu')
                self.vision_dim = 768
            clip_weight = clip_weight.state_dict()

            self.vision_encoder = build_model(clip_weight, self.config.vision_resolution, self.config.checkpointing).float()
            
        else:
            ## 
            from transformers import CLIPImageProcessor, CLIPModel
            self.vision_encoder = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            self.image_processor = CLIPImageProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            self.vision_dim = 1024 # ViT-H-14 has 1280 hidden size
            self.vision_encoder.requires_grad_(False)  # Freeze the vision encoder
    


    def forward_vision_encoder(self, images):
        img_inputs = self.image_processor(images=images, return_tensors="pt").to(torch.device('cuda'))
        
        #vision_output = self.vision_encoder.get_image_features(**img_inputs)
        vision_output = self.vision_encoder.vision_model(**img_inputs)  # [N, 1280]


        return vision_output  

    def forward_eeg_encoder(self, eeg_data, subject):
        if self.config.eeg_encoder_type == 'gwitvec':
            eeg_output = self.eeg_encoder(eeg_data, subject, return_vector=True)
        elif self.config.eeg_encoder_type == 'gwit':
            eeg_output = self.eeg_encoder(eeg_data, subject, return_vector=False)
        elif self.config.eeg_encoder_type == 'lstm':
            eeg_data = eeg_data.view(-1,eeg_data.shape[2],eeg_data.shape[1])
            eeg_output = self.eeg_encoder(eeg_data)[1]
        elif self.config.eeg_encoder_type == 'lstm_cross_attention':
            eeg_data = eeg_data.view(-1,eeg_data.shape[2],eeg_data.shape[1])
            eeg_output = self.eeg_encoder(eeg_data) # in this case it should return just the feat
        elif self.config.eeg_encoder_type == 'lstm_pretrained':
            eeg_data = eeg_data.view(-1,eeg_data.shape[2],eeg_data.shape[1])
            eeg_output = self.eeg_encoder(eeg_data)[0] #The lstm eeg encoder's forward method returns a tuple, where 0 is the pooled feature with projection and normalization
        elif self.config.eeg_encoder_type == 'ATMS' or self.config.eeg_encoder_type == 'ATMS_SDenc':
            eeg_output = self.eeg_encoder(eeg_data, subject) # [b,1024]
        elif self.config.eeg_encoder_type == 'ATMS_crossatt':
            eeg_output = self.eeg_encoder(eeg_data, subject) # [b,1024]
        else:
            raise NotImplementedError
        return eeg_output

## forward text encoder is implicitly implemented in gram.py batch_get function


    def pool_vision_for_contra(self, feature):  #feature b ,n ,x ,c
        #### always use frame_avg  for retrieval
        if self.config.vision_encoder_type.startswith('clip') or self.config.vision_encoder_type.startswith('evaclip'):
            feature = feature[:,:,0]
        elif self.config.vision_encoder_type.startswith('swin'):
            feature = feature.mean(dim=2)
        elif self.config.eeg_encoder_type == 'ATMS_crossatt':
            pooled_feature = feature.pooler_output
            feature = self.vision_encoder.visual_projection(pooled_feature)
        else:
            #feature = feature.last_hidden_state.mean(dim=1) # pooler output?
            visual_hidden = feature.pooler_output  # [N, 1280]
            feature = self.vision_encoder.visual_projection(visual_hidden)
        #feature = torch.mean(feature, dim=1)
        return feature

    def pool_text_for_contra(self, feature, attention_mask):  #feature b ,n ,x, c
        if self.config.eeg_encoder_type == 'ATMS_crossatt':
            pooled_text_feature = feature.pooler_output #could (should?) use it for lstm too..
            pooled_text_feature = self.vision_encoder.text_projection(pooled_text_feature)
        else:
            hidden = feature.last_hidden_state
            #Use end-of-text token for pooling, but take padding into account
            lengths = attention_mask.long().sum(dim=1) - 1
            idx = torch.arange(hidden.size(0), device = hidden.device)
            eot_token = hidden[idx, lengths] #should be [b, 1024]
            pooled_text_feature = self.vision_encoder.text_projection(eot_token)
        return pooled_text_feature

    def pool_eeg_for_contra(self, feature):
        if self.config.eeg_encoder_type == 'gwitvec': # [b, 2560, 64]
            pooled_feature = feature.mean(dim=2) 
        elif self.config.eeg_encoder_type =='gwit': # [b, 320,64,64]
            pooled_feature = feature.mean(dim=[2,3])
        elif self.config.eeg_encoder_type == 'lstm':
            pooled_feature = feature
        elif self.config.eeg_encoder_type == 'lstm_cross_attention':
            pooled_feature = feature
        elif self.config.eeg_encoder_type == 'lstm_pretrained':
            pooled_feature = feature
        elif self.config.eeg_encoder_type == 'tsconv': # [b, 2960]
            pooled_feature = feature
        elif self.config.eeg_encoder_type == 'ATMS' or 'ATMS_crossatt': # [b, 1024]
            pooled_feature = feature
        else: 
            raise NotImplementedError
        #pooled_feature = torch.mean(pooled_feature, dim=1)
        return pooled_feature


    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def get_multimodal_forward_input_vision(self, vision_output):

        #b,n,x,c = vision_output.shape
        b,n, x = vision_output.shape
        vision_output = self.hidden_trans_vision_multimodal(vision_output)  
        #print("vision multimodal after transformation: ", vision_output.shape)

        #if self.config.frame_embedding_type == 'adaptive':
        #    if n!=self.vision_frame_embedding.shape[1]: #### testing and interpolate
        #        # dtype = self.vision_frame_embedding.dtype
        #        vision_frame_embedding = F.interpolate(self.vision_frame_embedding.float().permute(0,2,1),n,mode='nearest').permute(0,2,1).to(self.vision_frame_embedding)
        #    else:
        #        vision_frame_embedding = self.vision_frame_embedding
#
#
        #    vision_output =  vision_output + vision_frame_embedding.unsqueeze(-2)
#
#
#
        #elif self.config.frame_embedding_type == 'none':
        #    pass

        #vision_output =  vision_output.reshape(b,-1,self.eeg_dim) 
        vision_output =  vision_output + self.vision_type_embeddings

        
        return vision_output

    def get_multimodal_forward_input_eeg(self, eeg_output):
        
        if self.config.eeg_encoder_type == 'gwitvec': #[b, 64, 2560]
            b, F, T = eeg_output.shape
            #print("EEG OUTPUT SHAPE: ", eeg_output.shape)
            #eeg_reshaped = eeg_output.permute(0, 2, 1).reshape(-1, F) # [b*64, 2560]
            eeg_permute = eeg_output.view(-1,eeg_output.shape[2],eeg_output.shape[1])
            eeg_transformed = self.hidden_trans_eeg_multimodal(eeg_permute)
            #print("EEG TRANSOFRMED SHAPE: ", eeg_transformed.shape)
            eeg_out = eeg_transformed.reshape(b, -1, self.multimodal_dim)
            #print("EEG OUT SHAPE AFTER RESHAPE: ", eeg_out.shape)
            eeg_out = eeg_out + self.eeg_type_embeddings

        elif self.config.eeg_encoder_type == 'gwit': #[b, 320,64,64]
            b, c, h, w = eeg_output.shape
            # Flatten spatial dimensions: [b, 320, 64, 64] -> [b, 320, 64]
            eeg_flattened = eeg_output.view(b, c, -1)
            # Permute to match expected input format: [b, 64, 320]
            eeg_permute = eeg_flattened.view(-1,eeg_output.shape[2],eeg_output.shape[1])
            # Apply hidden transformation: [b, 64, 320] -> [b, 64, multimodal_dim]
            eeg_transformed = self.hidden_trans_eeg_multimodal(eeg_permute)
            # Reshape for output: [b, 64, multimodal_dim]
            eeg_out = eeg_transformed.reshape(b, -1, self.multimodal_dim)
            # Add type embeddings
            eeg_out = eeg_out + self.eeg_type_embeddings

        #TODO Is this correct?
        elif self.config.eeg_encoder_type == 'lstm' or self.config.eeg_encoder_type =='lstm_pretrained' or self.config.eeg_encoder_type == 'lstm_cross_attention': #[b, 128, 64]
            b, c = eeg_output.shape
            eeg_transformed = self.hidden_trans_eeg_multimodal(eeg_output)  # eeg_output is already in the right shape
            eeg_out = eeg_transformed.reshape(b, -1, self.multimodal_dim)
            eeg_out = eeg_out + self.eeg_type_embeddings
        return eeg_out

    def get_multimodal_forward_input_text(self, text_output):
        #print("text multimodal befor transformation: ", text_output.shape)
        text_output = self.hidden_trans_text_multimodal(text_output) #do i need it if i am already 1024-d?
        #print("text multimodal after transformation: ", text_output.shape)
        #text_output = text_output.reshape(text_output.shape[0], -1, self.eeg_dim)  # Reshape to [B, N, C]
        text_output = text_output + self.text_type_embeddings
        return text_output
