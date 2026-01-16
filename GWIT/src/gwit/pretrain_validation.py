import torch
import math
from tqdm import tqdm
import wandb
from utils.distributed import concat_all_gather, all_gather_list
from gram_utils import volume_computation3
from datasets import load_dataset
import random
import numpy as np
from transformers import AutoTokenizer
from torchvision import transforms
import os
import torch.distributed as dist
from accelerate.logging import get_logger
from dataset_EEG.name_map_ID import id_to_caption
import torch.nn.functional as F
import hashlib

from utils.logger import LOGGER
from transformers import AutoTokenizer
LOGGER = get_logger(__name__)

def compute_metric_ret(score_matrix, ids, ids_txt, direction='forward'):

    assert score_matrix.shape == (len(ids_txt),len(ids))

    if direction == 'forward': ### text-to-vision retrieval
        indice_matrix = score_matrix.sort(dim=-1,descending=True)[1].tolist()
        rank = []
        for i in range(len(ids_txt)):
            # gt_indice = ids.index(ids_txt[i][0])
            gt_indice = ids.index(ids_txt[i])
            rank.append(indice_matrix[i].index(gt_indice))
        
        rank = torch.tensor(rank).to(score_matrix)
        
        vr_r1 = (rank < 1).sum().item() / len(ids_txt)
        vr_r5 = (rank < 5).sum().item() / len(ids_txt)
        vr_r10 = (rank < 10).sum().item() / len(ids_txt)
        v_medianR = torch.median(rank).item() +1
        v_meanR = torch.mean(rank).item() +1
 
        eval_log = {'forward_r1': round(vr_r1*100,1),
                    'forward_recall': f'{round(vr_r1*100,1)}/{round(vr_r5*100,1)}/{round(vr_r10*100,1)}',
                    'forward_ravg': round((vr_r1 + vr_r5 + vr_r10)/3 *100,1)
                   }
   
    else: ### vision-to-text retrieval
       
        indice_matrix = score_matrix.sort(dim=0,descending=True)[1].permute(1,0).tolist()
        rank = []
        for i in range(len(ids)):
            gt_indices=[]
            for idx, id in enumerate(ids_txt):
                if id == ids[i]:
                    gt_indices.append(idx)

            rank.append(min([indice_matrix[i].index(idx) for idx in gt_indices]))
        
        rank = torch.tensor(rank).to(score_matrix)
        
        tr_r1 = (rank < 1).sum().item() / len(ids)
        tr_r5 = (rank < 5).sum().item() / len(ids)
        tr_r10 = (rank < 10).sum().item() / len(ids)
        t_medianR = torch.median(rank).item() +1
        t_meanR = torch.mean(rank).item() +1

        eval_log = {
                    'backward_r1': round(tr_r1*100,1),
                    'backward_recall': f'{round(tr_r1*100,1)}/{round(tr_r5*100,1)}/{round(tr_r10*100,1)}',
                    'backward_ravg': round((tr_r1 + tr_r5 + tr_r10)/3 *100,1)
                  }
    

    return eval_log

def compute_metric_ret_area(score_matrix, ids, ids_txt, direction='forward'):

    assert score_matrix.shape == (len(ids_txt),len(ids))

    if direction == 'forward': ### text-to-vision retrieval
        indice_matrix = score_matrix.sort(dim=-1,descending=False)[1].tolist()
        rank = []
        for i in range(len(ids_txt)):
            # gt_indice = ids.index(ids_txt[i][0])
            gt_indice = ids.index(ids_txt[i])
            rank.append(indice_matrix[i].index(gt_indice))
        
        rank = torch.tensor(rank).to(score_matrix)
        
        vr_r1 = (rank < 1).sum().item() / len(ids_txt)
        vr_r5 = (rank < 5).sum().item() / len(ids_txt)
        vr_r10 = (rank < 10).sum().item() / len(ids_txt)
        v_medianR = torch.median(rank).item() +1
        v_meanR = torch.mean(rank).item() +1
 
        eval_log = {'forward_r1': round(vr_r1*100,1),
                    'forward_recall': f'{round(vr_r1*100,1)}/{round(vr_r5*100,1)}/{round(vr_r10*100,1)}',
                    'forward_ravg': round((vr_r1 + vr_r5 + vr_r10)/3 *100,1)
                   }
   
    else: ### vision-to-text retrieval
       
        indice_matrix = score_matrix.sort(dim=0,descending=False)[1].permute(1,0).tolist()
        rank = []
        for i in range(len(ids)):
            gt_indices=[]
            for idx, id in enumerate(ids_txt):
                if id == ids[i]:
                    gt_indices.append(idx)

            rank.append(min([indice_matrix[i].index(idx) for idx in gt_indices]))
        
        rank = torch.tensor(rank).to(score_matrix)
        
        tr_r1 = (rank < 1).sum().item() / len(ids)
        tr_r5 = (rank < 5).sum().item() / len(ids)
        tr_r10 = (rank < 10).sum().item() / len(ids)
        t_medianR = torch.median(rank).item() +1
        t_meanR = torch.mean(rank).item() +1

        eval_log = {
                    'backward_r1': round(tr_r1*100,1),
                    'backward_recall': f'{round(tr_r1*100,1)}/{round(tr_r5*100,1)}/{round(tr_r10*100,1)}',
                    'backward_ravg': round((tr_r1 + tr_r5 + tr_r10)/3 *100,1)
                  }

    return eval_log


@torch.no_grad()
def evaluate_pretraining(model, args, global_step, accelerator, eval_dataset, eval_dataloader):
    """
    Evaluate on text, image, eeg alignments:
      - gramian volume
      - mean cosine(text,image), cosine(text,eeg), cosine(image,eeg)
    Logs to wandb and returns a flat dict of scalars.
    """

    model.eval()
    ids = []
    ids_txt = []
    input_ids = []
    attention_mask = []

    feat_t = []
    feat_i = []
    feat_e = []


    progress_bar = tqdm(
        range(0, math.ceil(len(eval_dataset)/args.train_batch_size)),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    for step, batch in enumerate(eval_dataloader):
        feat_t.append(model.batch_get(batch, 'feat_t'))
        feat_i.append(model.batch_get(batch, 'feat_i'))
        feat_e.append(model.batch_get(batch, 'feat_e'))

        input_ids.append(model.batch_get(batch ,'input_ids'))
        attention_mask.append(model.batch_get(batch ,'attention_mask'))
        
        ids += batch["ids"]
        ids_txt += batch["ids_txt"]

        progress_bar.update(1)
        progress_bar.set_postfix()

    # concat + gather across devices
    #ids = [j for i in all_gather_list(ids) for j in i]
    #ids_txt = [j for i in all_gather_list(ids_txt) for j in i]

    input_ids = torch.cat([i for i in input_ids], dim=0)
    input_ids = concat_all_gather(input_ids)
    attention_mask = torch.cat([i for i in attention_mask], dim=0)
    attention_mask = concat_all_gather(attention_mask)

    feat_t = torch.cat(feat_t, dim=0)
    feat_i = torch.cat(feat_i, dim=0)
    feat_e = torch.cat(feat_e, dim=0)

    feat_t = concat_all_gather(feat_t)
    feat_i = concat_all_gather(feat_i)
    feat_e = concat_all_gather(feat_e)


    # 3) gramian volume
    #    shape [N,N] → take min over columns per row, then mean
    vol = volume_computation3(feat_t, feat_i, feat_e)
    min_values_volume = vol.min(dim=1).values
    mean_values_volume = torch.mean(min_values_volume).item()
    std_min_vol  = min_values_volume.std().item()

    ret_area_forward = compute_metric_ret_area(vol, ids, ids_txt, direction='forward')
    ret_area_forward = {k.replace('forward', 'volume_T2D'): v for k, v in ret_area_forward.items()}
    ret_area_backward = compute_metric_ret_area(vol.T, ids, ids_txt, direction='forward')
    ret_area_backward = {k.replace('backward', 'volume_D2T'): v for k, v in ret_area_backward.items()}


    cosine_TI = torch.matmul(feat_t, feat_i.permute(1,0))
    cosine_TI = compute_metric_ret(cosine_TI, ids, ids_txt, direction='forward')
    
    cosine_IT = torch.matmul(feat_i, feat_t.permute(1,0))
    cosine_IT = compute_metric_ret(cosine_IT, ids, ids_txt, direction='forward')

    cosine_TE = torch.matmul(feat_t, feat_e.permute(1,0))
    cosine_TE = compute_metric_ret(cosine_TE, ids, ids_txt, direction='forward')

    cosine_ET = torch.matmul(feat_e, feat_t.permute(1,0))
    cosine_ET = compute_metric_ret(cosine_ET, ids, ids_txt, direction='forward')


    ## compute itc_score
    #   t-i
    score_matrix_TI = torch.matmul(feat_t, feat_i.permute(1,0))
    log = compute_metric_ret(score_matrix_TI, ids, ids_txt, direction='forward')
    log = {k.replace('forward', 'TI'): v for k, v in log.items()}

    #   t-e
    score_matrix_TE = torch.matmul(feat_t, feat_e.permute(1,0))
    log_2 = compute_metric_ret(score_matrix_TE, ids, ids_txt, direction='forward')
    log_2 = {k.replace('forward', 'TE'): v for k, v in log_2.items()}

    # 5) log & return
    metrics = {
        "gramian/value":    mean_values_volume,
        "gramian/std":      std_min_vol,
        "ret_area/forward":  ret_area_forward,
        "ret_area/backward": ret_area_backward,
        "cosine/TI":        cosine_TI,
        "cosine/IT":        cosine_IT,
        "cosine/TE":        cosine_TE,
        "cosine/ET":        cosine_ET,
        "ret_itc_TI": log,
        "ret_ITC_TE": log_2,
    }
    print(metrics)
    if accelerator.is_main_process:
        wandb.log(metrics, step=global_step)

    return metrics