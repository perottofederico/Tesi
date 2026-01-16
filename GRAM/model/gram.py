import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#os.environ["TRANSFORMERS_OFFLINE"] = "1"
import json
import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from utils.logger import LOGGER
from .general_module import TokenMasker, MMGeneralModule, Contra_head, Match_head
from utils.distributed import all_gather_with_grad, concat_all_gather, all_gather_list
from torch.nn import LayerNorm as LayerNorm
from easydict import EasyDict as edict
from utils.volume import volume_computation4,volume_computation3, volume_computation5

class GRAM(MMGeneralModule):
    def __init__(self, config):
        super().__init__()
    
        self.config = config
        
        print(self.config)
        self.eeg_encoder = self.config.eeg_encoder_type
        self.dataset = self.config.dataset

        self.construct_vision_encoder()
        self.load_clip_text_encoder()
        self.construct_eeg_encoder()
        

        contra_dim = self.config.contra_dim
        print("Contra dim: ", contra_dim)
        print("multimodal_dim: ", self.multimodal_dim)
        print("vision_dim: ", self.vision_dim)
        print("eeg_dim: ", self.eeg_dim)
        self.contra_head_t = Contra_head(self.multimodal_dim, contra_dim)
        self.contra_head_v = Contra_head(self.vision_dim, contra_dim)
        self.contra_temp = nn.Parameter(torch.tensor(0.07))  #nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) #0.07
    
        if (self.config.eeg_encoder_type != 'ATMS_crossatt'):
            self.itm_head = Match_head(self.eeg_dim)
            self.hidden_trans_vision_multimodal = nn.Sequential(nn.Linear(1280, self.eeg_dim),LayerNorm(self.eeg_dim, eps=1e-12))
            self.hidden_trans_text_multimodal = nn.Sequential(nn.Linear(self.multimodal_dim, self.eeg_dim),LayerNorm(self.eeg_dim, eps=1e-12))
            self.text_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.eeg_dim))
            self.vision_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.eeg_dim))
        else:
            self.itm_head = Match_head(contra_dim)
            self.hidden_trans_vision_multimodal = nn.Sequential(nn.Linear(1280, self.eeg_dim),LayerNorm(self.eeg_dim, eps=1e-12))
            self.hidden_trans_text_multimodal = nn.Sequential(nn.Linear(self.multimodal_dim, self.eeg_dim),LayerNorm(self.eeg_dim, eps=1e-12))
            self.text_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.eeg_dim))
            self.vision_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.eeg_dim))
        self.itm_ratio = config.itm_ratio
        self.max_caption_len = config.max_caption_len

        self.contra_head_eeg = Contra_head(self.eeg_dim, contra_dim)
        self.hidden_trans_eeg_multimodal = nn.Sequential(nn.Linear(self.eeg_dim, self.multimodal_dim),LayerNorm(self.multimodal_dim, eps=1e-12))
        self.eeg_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim))


    def batch_get(self, batch, key):
        if key in batch:
            return batch[key]

        
        # EEG related data
        elif key == 'subject':
            batch[key] = batch['eeg_subjects']

        #used only when computing DAM loss with cross attention capable encoders
        elif key == 'eeg_data':
            eeg_data = batch['conditioning_pixel_values']
            if self.dataset == "EEGCVPR":
                eeg_data = eeg_data.view(-1,eeg_data.shape[2],eeg_data.shape[1]).to(torch.device('cuda'))
            batch[key] = eeg_data
        
        elif key == 'eeg_output':
            subject = self.batch_get(batch, 'subject')
            eeg_output = self.forward_eeg_encoder(batch.conditioning_pixel_values, subject)
            batch[key] = eeg_output


        # Caption / Text related data
        elif key == 'caption_tokens':
            caption_tokens = self.tokenizer(batch.raw_captions,
                                            padding="max_length",
                                            truncation=True,
                                            max_length=self.max_caption_len,
                                            return_tensors="pt").to(torch.device('cuda'))
            batch[key] = caption_tokens

        elif key == 'caption_output':
            caption_tokens = self.batch_get(batch, 'caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_output = self.text_encoder(input_ids = input_ids, attention_mask = attention_mask)
            batch[key] = caption_output

        elif key == 'text_last_hidden_state':
            caption_output = batch.text_last_hidden_states


        # Vision / Image related data
        elif key == 'vision_output':
            vision_output = self.forward_vision_encoder(batch.images)
            batch[key] = vision_output
            
        elif key == 'image_last_hidden_state':
            batch[key] = vision_output.last_hidden_state

        elif key == 'condition_feats_v':
            if self.dataset == "EEGCVPR":
                vision_output = self.batch_get(batch, 'image_last_hidden_states')
            else: 
                vision_output = self.batch_get(batch, 'vision_output').last_hidden_state
            condition_feats_v = self.get_multimodal_forward_input_vision(vision_output)
            batch[key] = condition_feats_v
        
        elif key == 'condition_feats_t':
            if self.dataset == "EEGCVPR":
                caption_output = self.batch_get(batch, 'text_last_hidden_states')
            else:
                caption_output = self.batch_get(batch, 'caption_output').last_hidden_state
            condition_feats_t = self.get_multimodal_forward_input_text(caption_output)
            batch[key] = condition_feats_t
            
        elif key == 'condition_feats_e':
            eeg_output = self.batch_get(batch, 'eeg_output')
            condition_feats_e = self.get_multimodal_forward_input_eeg(eeg_output)
            batch[key] = condition_feats_e

        elif key == 'condition_feats_ve':
            condition_feats_v = self.batch_get(batch, 'condition_feats_v')
            condition_feats_e = self.batch_get(batch, 'condition_feats_e')
            condition_feats_ve = torch.cat((condition_feats_v, condition_feats_e),dim=1)
            batch[key] = condition_feats_ve

        elif key == 'condition_feats_vt':
            #print("[GRAM] Getting condition_feats_vt")
            condition_feats_v = self.batch_get(batch, 'condition_feats_v')
            condition_feats_t = self.batch_get(batch, 'condition_feats_t')
            condition_feats_vt = torch.cat((condition_feats_v, condition_feats_t),dim=1)
            batch[key] = condition_feats_vt

        elif key == 'condition_feats_vte':
            condition_feats_v = self.batch_get(batch, 'condition_feats_v')
            condition_feats_t = self.batch_get(batch, 'condition_feats_t')
            condition_feats_e = self.batch_get(batch, 'condition_feats_e')
            condition_feats_vte = torch.cat((condition_feats_v, condition_feats_t, condition_feats_e),dim=1)
            batch[key] = condition_feats_vte


        elif key == 'feat_v':
            if self.dataset == "EEGCVPR":
                feat_v = batch.image_features
            elif self.dataset == "THINGS":
                feat_v = batch['img_features']
            else:
                #base case
                vision_output = self.batch_get(batch, 'vision_output')
                feat_v = self.pool_vision_for_contra(vision_output)
                if self.config.contra_dim != 1024:
                    feat_v = self.contra_head_v(feat_v)
            feat_v = F.normalize(feat_v,dim=-1)
            batch[key] = feat_v

        elif key == 'feat_t':
            if self.dataset == "EEGCVPR" or self.dataset == "THINGS":
                feat_t = batch['text_features']
            else: 
                #default case
                caption_output = self.batch_get(batch, 'caption_output')
                attention_mask = self.batch_get(batch, 'caption_tokens').attention_mask
                feat_t = self.pool_text_for_contra(caption_output, attention_mask)
                if self.config.contra_dim != 1024:
                    feat_t = self.contra_head_t(feat_t)
            feat_t = F.normalize(feat_t,dim=-1)
            batch[key] = feat_t
        
        elif key == 'feat_e':
            '''
            If contra_dim is != eeg dim, I need to add a projection layer.
            note taht ATMS_crossatt eeg dim is 250 but it gets projected already to 1024 in the eeg encoder forward pass
            ''' 
            eeg_output = self.batch_get(batch, 'eeg_output')
            eeg_output_pooled = self.pool_eeg_for_contra(eeg_output)
            if self.config.eeg_encoder_type != 'ATMS_crossatt' or self.config.contra_dim != self.eeg_dim:
                eeg_output = self.contra_head_eeg(eeg_output_pooled)
            feat_e = F.normalize(eeg_output,dim=-1)
            batch[key] = feat_e
 
        return batch[key] 


    def forward(self, batch, task, compute_loss=True):
        batch = edict(batch)

        ### other datasets pretraining or finetuning
        output_ls = []
        task_ls = task.split('_')


        for task in task_ls:
            if task.startswith('ret'):
                ret_dict = self.forward_ret(batch, task, compute_loss=compute_loss)
                output_ls.append(ret_dict)

            else:
                raise NotImplementedError

        output_dict = {k:v for dic in output_ls for k,v in dic.items()  }
        return output_dict

    def compute_slice_scores(self, slice_multimodal_vision_input, slice_input_ids, slice_attention_mask):
            
        slice_output = self.multimodal_encoder.bert(input_ids = slice_input_ids,
                                                    attention_mask = slice_attention_mask,
                                                    encoder_hidden_states=slice_multimodal_vision_input).last_hidden_state
        #slice_output = self.text_encoder(input_ids = slice_input_ids, attention_mask = slice_attention_mask).last_hidden_state
        slice_scores = F.softmax(self.itm_head(slice_output[:,0]),dim=1)[:,1]

        return slice_scores


    def forward_ret(self, batch, task, compute_loss=True):
        if isinstance(batch.raw_captions[0],list): #### test
            batch.raw_captions = [i for j in batch.raw_captions for i in j]
        subtasks = task.split('%')[1:]
        if compute_loss:
            loss_dict={}
            loss_itc = []
            loss_itm = []
            loss_area = []

            #Extract text features
            feat_t = self.batch_get(batch,'feat_t')
            feat_t_all = concat_all_gather(feat_t)
            #Extract visual features
            feat_v = self.batch_get(batch,'feat_v')
            feat_v_all = concat_all_gather(feat_v)
            #Extract eeg features
            feat_e = self.batch_get(batch,'feat_e')
            feat_e_all = concat_all_gather(feat_e)

            #caption_tokens = self.batch_get(batch, 'caption_tokens')
            #input_ids, attention_mask = caption_tokens.input_ids, caption_tokens.attention_mask
            #input_ids_collate = concat_all_gather(input_ids)
            #attention_mask_collate = concat_all_gather(attention_mask)
            if(self.config.eeg_encoder_type == 'lstm_cross_attention' or "ATMS_crossatt"):
                #only used to build negative samples when computing DAM loss
                eeg_data = self.batch_get(batch, 'eeg_data')
                eeg_data_collate = concat_all_gather(eeg_data)


            #       VOLUME LOSS COMPUTATION
            #           VOLUME ITC

            #Volume (Text, batch_all)
            #volume = volume_computation3(feat_t,feat_v_all,feat_e_all)
            volume = volume_computation3(feat_e, feat_v_all, feat_t_all)
            volume = volume / self.contra_temp
            #AreaT (Video,batch_all)
            volumeT = volume_computation3(feat_e_all, feat_v, feat_t).T
            volumeT = volumeT / self.contra_temp
            
            rank = dist.get_rank()
            bs = feat_t.size(0)
            
            targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(volume.device)

            loss = (
                    F.cross_entropy(-volume, targets, label_smoothing=0.1) #d2a
                    + F.cross_entropy(-volumeT, targets, label_smoothing=0.1) #a2d
            ) / 2

            loss_area.append(loss)


            #   AREA VID ITM 
            #vid_sim = feat_t @ feat_v_all.T
            #vid_simT = feat_v @ feat_t_all.T
            if self.config.eeg_encoder_type == 'lstm_cross_attention' or self.config.eeg_encoder_type == "ATMS_crossatt":
                condition_feats = self.batch_get(batch, f'condition_feats_vt')
                condition_feats_collate = all_gather_with_grad(condition_feats)

                with torch.no_grad():
                    weights_eeg2cond = F.softmax(-(volume), dim=1) + 1e-4
                    weights_eeg2cond[:, rank * bs : rank * bs + bs].fill_diagonal_(0)
                    weights_cond2eeg = F.softmax(-(volumeT), dim=1) + 1e-4
                    weights_cond2eeg[:, rank * bs : rank * bs + bs].fill_diagonal_(0)

                condition_feats_neg = []
                for b in range(bs): 
                    neg_idx = torch.multinomial(weights_eeg2cond[b], 1).item()
                    condition_feats_neg.append(condition_feats_collate[neg_idx])
                condition_feats_neg = torch.stack(condition_feats_neg, dim=0)

                eeg_neg = []
                for b in range(bs):
                    neg_idx = torch.multinomial(weights_cond2eeg[b], 1).item()
                    eeg_neg.append(eeg_data_collate[neg_idx]) 

                eeg_neg = torch.stack(eeg_neg, dim=0)

                eeg_inputs = torch.cat((eeg_data, eeg_data, eeg_neg),dim=0)
                condition_feats = torch.cat((condition_feats,condition_feats_neg,condition_feats),dim=0)

                #output = self.multimodal_encoder.bert(input_ids = input_ids_1,
                #                            attention_mask = attention_mask_1,
                #                            encoder_hidden_states=condition_feats).last_hidden_state
                if self.config.eeg_encoder_type == "ATMS_crossatt":
                    #repeat subject ids three times for negative samples
                    subject_ids = self.batch_get(batch, 'subject')
                    output = self.eeg_encoder(eeg_inputs, subject_ids =  subject_ids.repeat(3), conditioning = condition_feats)#[0]
                else:
                    output = self.eeg_encoder(eeg_inputs, conditioning = condition_feats)[0]

                batch_size = condition_feats_neg.shape[0]
                logits = self.itm_head(output)
                ground_truth = torch.zeros(batch_size*3).long().cuda()
                ground_truth[:batch_size] = 1
                loss = F.cross_entropy(logits,ground_truth) #itm (dtm)
                loss_itm.append(self.itm_ratio * loss)

            

            for task in subtasks:

                loss_itc.append(torch.tensor(0))#*loss)
                loss_itm.append(torch.tensor(0))#*(self.itm_ratio * loss))


            loss_itc = sum(loss_itc)/len(loss_itc)
            loss_dict['loss_itc'] = loss_itc          
            loss_itm = sum(loss_itm)/len(loss_itm)
            loss_dict['loss_itm'] = loss_itm
            loss_area = sum(loss_area)/len(loss_area)
            loss_dict['loss_area'] = loss_area
            loss_dict['temp'] = self.contra_temp
            return loss_dict
          
        else:

            evaluation_dict = {}
            feat_t = self.batch_get(batch,'feat_t')
            evaluation_dict['feat_t'] = feat_t 

            feat_v = self.batch_get(batch,'feat_v')
            evaluation_dict['feat_v'] = feat_v

            feat_e = self.batch_get(batch,'feat_e')
            evaluation_dict['feat_e'] = feat_e

            '''
            if self.config.eeg_encoder_type == ('lstm_cross_attention' or "ATMS_crossatt"):
                caption_tokens = self.batch_get(batch,'caption_tokens')
                evaluation_dict['input_ids'] = caption_tokens.input_ids
                evaluation_dict['attention_mask'] = caption_tokens.attention_mask
                
                for task in subtasks:
                    #### compute_itc
                    #assert task in ['tv','ta','tva','tvs','tvas','tvasd']
                    # feat_cond = self.batch_get(batch,f'feat_{task[1:]}')
                    # evaluation_dict[f'feat_cond_{task}'] = feat_cond
                    condition_feats = self.batch_get(batch, f'condition_feats_{task[1:]}')
                    evaluation_dict[f'condition_feats_{task}'] = condition_feats
            '''
            return evaluation_dict
