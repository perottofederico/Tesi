import os
import json
import torch
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from utils.logger import LOGGER
from utils.distributed import  all_gather_list, ddp_allgather
from utils.tool import NoOp
from easydict import EasyDict as edict
from utils.volume import volume_computation2, volume_computation4,volume_computation3, volume_computation5
import wandb
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

def evaluate_mm(model, val_dataloaders, run_cfg, global_step):

    eval_log = {}
    model.eval()
    for task, loader in val_dataloaders.items():
        LOGGER.info(f"evaluate on {task} task")
        val_log = evaluate_single(model, loader, task.split('--')[0], run_cfg, global_step,task.split('--')[1])
        eval_log[task] = val_log
    model.train()
    return eval_log

@torch.no_grad()
def evaluate_single(model, val_loader, task, run_cfg, global_step,dset_name):
    LOGGER.info("start running {} validation...".format(task))

    tasks = task.split('_')

    output_ls = []

    for task in tasks:
        if task.startswith('ret'):
            ret_dict = evaluate_ret(model, task, val_loader, global_step, run_cfg)
            #ret_dict = evaluate_ret_area(model, task, val_loader, global_step, run_cfg)
            output_ls.append(ret_dict)

    output_dict = {k:v for dic in output_ls for k,v in dic.items() }
    return output_dict

@torch.no_grad()
def evaluate_ret(model, tasks, val_loader, global_step, run_cfg):
    val_log = {}
    ids = []
    ids_txt = []
    #input_ids = []
    #attention_mask = []

    # additional data for t-SNE
    subjects_list = []
    # captions_list = []
 
    subtasks = tasks.split('%')[1:]
    store_dict = {}
    feat_t = []
    feat_a = []
    feat_e = []
    feat_v = []
    feat_s = []
    feat_d = []

    images = []
    labels = []

    for task in subtasks:
        store_dict[f'condition_feats_{task}'] = []        

    for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
        batch = edict(batch)
        images.append(batch.vision_pixels)
        labels.append(batch.labels)

        evaluation_dict= model(batch, tasks, compute_loss=False)

        feat_t.append(evaluation_dict['feat_t'])
        feat_v.append(evaluation_dict['feat_v'])
        feat_e.append(evaluation_dict['feat_e'])
        if 'feat_s' in evaluation_dict.keys():
            feat_s.append(evaluation_dict['feat_s'])
        if 'feat_d' in evaluation_dict.keys():
            feat_d.append(evaluation_dict['feat_d'])
      
        #input_ids.append(evaluation_dict['input_ids'])
        #attention_mask.append(evaluation_dict['attention_mask'])
        ids += batch.ids

        if 'ids_txt' in batch:
            if isinstance(batch['ids_txt'][0],list):
                ids_txt  +=  [j for i in batch.ids_txt for j in i]
            else:
                ids_txt  += batch.ids_txt
        else:    
            ids_txt  += batch.ids

        if 'eeg_subjects' in batch:
            subjects_list.extend(batch['eeg_subjects'].cpu().numpy().tolist())
        #if 'raw_captions' in batch:
        #    captions_list.extend(batch['raw_captions'])
  
        #for task in subtasks:
        #    # store_dict[f'feat_cond_{task}'].append(evaluation_dict[f'feat_cond_{task}'])    
        #    store_dict[f'condition_feats_{task}'].append(evaluation_dict[f'condition_feats_{task}'])

        
            
    ids = [j for i in all_gather_list(ids) for j in i]
    ids_txt = [j for i in all_gather_list(ids_txt) for j in i]
    #input_ids = torch.cat([i for i in input_ids],dim=0)
    #input_ids = ddp_allgather(input_ids)
    #attention_mask = torch.cat([i for i in attention_mask],dim=0)
    #attention_mask = ddp_allgather(attention_mask)
        
    feat_t = torch.cat(feat_t, dim = 0)
    feat_t = ddp_allgather(feat_t)

    #feat_a = torch.cat(feat_a, dim = 0)
    #feat_a = ddp_allgather(feat_a)
    feat_e = torch.cat(feat_e, dim = 0)
    feat_e = ddp_allgather(feat_e)

    feat_v = torch.cat(feat_v, dim = 0)
    feat_v = ddp_allgather(feat_v)

    if len(feat_s)>0:
        feat_s = torch.cat(feat_s, dim = 0)
        feat_s = ddp_allgather(feat_s)

    if len(feat_d)>0:
        feat_d = torch.cat(feat_d, dim = 0)
        feat_d = ddp_allgather(feat_d)

    # gather step for the additional data
    subjects_list = [j for i in all_gather_list(subjects_list) for j in i]
    #captions_list = [j for i in all_gather_list(captions_list) for j in i]
    
    area = volume_computation3(feat_e,feat_v,feat_t)

    min_values_volume = torch.min(area, 1).values
    mean_values_volume = torch.mean(min_values_volume)
    val_log[f"gramian_value"] = {"value": mean_values_volume.item()}
    
    #Compute retrieval metrics based on volume, both forward (anchor to data) and backward (Data to anchor)
    log = compute_metric_ret_area(area, ids, ids_txt, direction='forward')
    log = {k.replace('forward','volume_T2D'): v for k,v in log.items()}

    val_log[f'ret_area_forward'] = log

    log = compute_metric_ret_area(area.T, ids, ids_txt, direction='forward')
    log = {k.replace('backward','volume_D2T'): v for k,v in log.items()}

    val_log[f'ret_area_backward'] = log

    #### compute itm_score metrics using volume,
    #### commented out because most models don't use itm loss
    #store_dict[f'condition_feats_{task}'] = torch.cat(store_dict[f'condition_feats_{task}'],dim=0)
    #itm_rerank_num = model.config.itm_rerank_num
    #score_matrix = refine_score_matrix(store_dict[f'condition_feats_{task}'], input_ids, attention_mask, -area , model, itm_rerank_num, direction='forward')#-(area-video_similarity)
    #log = compute_metric_ret(score_matrix, ids, ids_txt, direction='forward')
    #log = {k.replace('forward','volume_ITM_T2D'): v for k,v in log.items()}
    #score_matrix = refine_score_matrix(store_dict[f'condition_feats_{task}'], input_ids, attention_mask, -area, model, itm_rerank_num, direction='backward') #-(area-video_similarity)
    #log2 = compute_metric_ret(score_matrix, ids, ids_txt, direction='backward')
    #log2 = {k.replace('backward','volume_ITM_D2T'): v for k,v in log2.items()}
    #log.update(log2)
    #val_log[f'ret_itm_area'] = log
    
    # compute retrieval using cosine similarity
    # TV and VT are sanity checks 
    cosine_TV = torch.matmul(feat_t, feat_v.permute(1,0))
    cosine_TV = compute_metric_ret(cosine_TV, ids, ids_txt, direction='forward')
    val_log[f'cosine_TV'] = cosine_TV
    
    cosine_VT = torch.matmul(feat_v, feat_t.permute(1,0))
    cosine_VT = compute_metric_ret(cosine_VT, ids, ids_txt, direction='forward')
    val_log[f'cosine_VT'] = cosine_VT
    
    cosine_TE = torch.matmul(feat_t, feat_e.permute(1,0))
    cosine_TE = compute_metric_ret(cosine_TE, ids, ids_txt, direction='forward')
    val_log[f'cosine_TE'] = cosine_TE

    cosine_ET = torch.matmul(feat_e, feat_t.permute(1,0))
    cosine_ET = compute_metric_ret(cosine_ET, ids, ids_txt, direction='forward')
    val_log[f'cosine_ET'] = cosine_ET

    cosine_VE = torch.matmul(feat_v, feat_e.permute(1,0))
    cosine_VE = compute_metric_ret(cosine_VE, ids, ids_txt, direction='forward')
    val_log[f'cosine_VE'] = cosine_VE

    cosine_EV = torch.matmul(feat_e, feat_v.permute(1,0))
    cosine_EV = compute_metric_ret(cosine_EV, ids, ids_txt, direction='forward')
    val_log[f'cosine_EV'] = cosine_EV

    ## compute itc_score
    # this is just retrieval based on cosine similarity again...
    print("computing itc_score")
    for task in subtasks:
        print("-task: ", task)
        if  task == "tvas" or task == "tva"or task == "evt":
            continue
        if task=='tv':
            score_matrix_t_cond = torch.matmul(feat_t, feat_v.permute(1,0))
        elif task=='te':
            score_matrix_t_cond = torch.matmul(feat_t, feat_e.permute(1,0))
        elif task == 'et':
            score_matrix_t_cond = torch.matmul(feat_e, feat_t.permute(1,0))
        elif task == 'ev':
            score_matrix_t_cond = torch.matmul(feat_e, feat_v.permute(1,0))
        store_dict[f'score_matrix_t_cond_{task}'] = score_matrix_t_cond
        log = compute_metric_ret(score_matrix_t_cond, ids, ids_txt, direction='forward')
        log = {k.replace('forward','video'): v for k,v in log.items()}
        if model.config.ret_bidirection_evaluation:
            log2 = compute_metric_ret(score_matrix_t_cond, ids, ids_txt, direction='backward')
            log2 = {k.replace('backward','txt'): v for k,v in log2.items()}
            log.update(log2)

        val_log[f'ret_itc_{task}'] = log


    #### compute itm_score
    #### itm_score again, using cosine similarity. Commented out because most models don't use itm loss
    #print("computing itm_score")
    #for task in subtasks:
    #    print("-task: ", task)
    #    if  task == "tvas" or task == "tva" or task == "evt":
    #        continue
    #    if task!="tvas" and task!="tva":
    #        store_dict[f'condition_feats_{task}'] = torch.cat(store_dict[f'condition_feats_{task}'],dim=0)
    #    itm_rerank_num = model.config.itm_rerank_num
    #    score_matrix = refine_score_matrix(store_dict[f'condition_feats_{task}'], input_ids, attention_mask, store_dict[f'score_matrix_t_cond_{task}'], model, itm_rerank_num, direction='forward')
    #    log = compute_metric_ret(score_matrix, ids, ids_txt, direction='forward')
    #    log = {k.replace('forward','video'): v for k,v in log.items()}
#
    #    if model.config.ret_bidirection_evaluation:
    #        score_matrix = refine_score_matrix(store_dict[f'condition_feats_{task}'], input_ids, attention_mask, store_dict[f'score_matrix_t_cond_{task}'], model, itm_rerank_num, direction='backward')
    #        log2 = compute_metric_ret(score_matrix, ids, ids_txt, direction='backward')
    #        log2 = {k.replace('backward','txt'): v for k,v in log2.items()}
    #        log.update(log2)
#
    #    val_log[f'ret_itm_{task}'] = log

    if dist.get_rank() == 0:
        wandb.log(val_log)


    #Retrieval visualization
    #create_retrieval_viz(
    #    feat_t,
    #    feat_v,
    #    feat_e,
    #    images, 
    #    labels, 
    #    ids, 
    #    ids_txt, 
    #    run_cfg, 
    #    step= "_eval" if run_cfg.mode == "testing" else global_step,
    #    modes = ('vc3', 'vc2_ev', 'vc2_et')
    #)
    
    
    #ATMS_acc = ATMS_accuracy(feat_t, feat_v, feat_e, labels, k=200)
    
    #Gramian values and "classic" retrieval by area (both volume_computation3 and volume_computation2, but with duplicates)
    LOGGER.info(evaluate_ret_area(feat_t, feat_v, feat_e, images, labels, ids, ids_txt, run_cfg, step=global_step))

    # Gallery should be deduped always, else it will hurt recall metrics
    print("="*50)
    print("Computing retrieval metrics with gallery AND query deduplication...")
    # Deduping both iamge gallery and the eeg queries
    results_both_deduped = compute_extended_retrieval_metrics(
        feat_t, feat_v, feat_e, images, labels, ids, ids_txt, subjects_list,
        dedupe_queries=True,
        output_dir=run_cfg.output_dir,
        file_prefix="both_deduped",
        step="_eval" if run_cfg.mode == "testing" else global_step 
    )
    print('\n')
    print("="*50)
    print("Computing retrieval metrics with ONLY gallery deduplication...")
    # Deduping only the image gallery
    results_gallery_deduped = compute_extended_retrieval_metrics(
        feat_t, feat_v, feat_e, images, labels, ids, ids_txt, subjects_list,
        dedupe_queries=False, 
        output_dir=run_cfg.output_dir,
        file_prefix="gallery_deduped",
        step="_eval" if run_cfg.mode == "testing" else global_step 
    )
    print("="*50)
    return val_log

#quick and simple version to compute gram volumes and retrieval metrics
@torch.no_grad()
def evaluate_ret_area(feat_t, feat_v, feat_e, images, labels, ids, ids_txt, run_cfg, step):
    val_log = {}
    area_evt = volume_computation3(feat_e,feat_v,feat_t)

    area_et = volume_computation2(feat_e,feat_t)
    area_ev = volume_computation2(feat_e,feat_v)

    min_values_volume = torch.min(area_evt, 1).values
    mean_values_volume = torch.mean(min_values_volume)
    val_log[f"gramian_value_evt"] = {"value": mean_values_volume.item()}
    min_values_volume = torch.min(area_et, 1).values
    mean_values_volume = torch.mean(min_values_volume)
    val_log[f"gramian_value_et"] = {"value": mean_values_volume.item()}
    min_values_volume = torch.min(area_ev, 1).values
    mean_values_volume = torch.mean(min_values_volume)
    val_log[f"gramian_value_ev"] = {"value": mean_values_volume.item()}

    met_evt = compute_metric_ret_area(area_evt, ids, ids_txt, direction='forward')
    met_et = compute_metric_ret_area(area_et, ids, ids_txt, direction='forward')
    met_ev = compute_metric_ret_area(area_ev, ids, ids_txt, direction='forward')

    val_log[f'ret_area_evt'] = {k.replace('forward','evt'): v for k,v in met_evt.items()}
    val_log[f'ret_area_et'] = {k.replace('forward','et'): v for k,v in met_et.items()}
    val_log[f'ret_area_ev'] = {k.replace('forward','ev'): v for k,v in met_ev.items()}

    return val_log

#unused
@torch.no_grad()
def refine_score_matrix(condition_feats, input_ids, attention_mask, score_matrix_t_cond, model, itm_rerank_num, direction='forward'):

    top_k = itm_rerank_num
    if direction=='forward':
        idxs = score_matrix_t_cond.topk(top_k,dim=1)[1]
    else:
        idxs = score_matrix_t_cond.topk(top_k,dim=0)[1]
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    nums = score_matrix_t_cond.shape[0]//world_size +1
    
    score_matrix_t_cond_new = torch.zeros_like(score_matrix_t_cond)
    idxs_new = torch.zeros_like(score_matrix_t_cond_new).long()
    if direction=='forward':
        for i in range(len(idxs)):
            for j in idxs[i]:
                idxs_new[i][j] = 1
    else:
        for i in range(idxs.shape[1]):
            for j in idxs[:,i]:
                idxs_new[j][i] = 1
    cur_length = condition_feats.shape[0]
    length_ls = all_gather_list(cur_length)
    start = 0
    start_ls = []
    end_ls = []
    for l in range(len(length_ls)):
        start_ls.append(start)
        end_ls.append(start+length_ls[l])
        start = start+length_ls[l]
    
    cur_score_matrix_t_cond = score_matrix_t_cond[:,start_ls[rank]:end_ls[rank]]
    cur_score_matrix_t_cond_new = score_matrix_t_cond_new[:,start_ls[rank]:end_ls[rank]]
    cur_idxs_new = idxs_new[:,start_ls[rank]:end_ls[rank]]

    if dist.get_rank() == 0:
        pbar = tqdm(total=cur_length)
    else:
        pbar = NoOp()
    for i in range(cur_length):
        if sum(cur_idxs_new[:,i] == 1) == 0:
            continue
        cur_scores = []
        cur_input_ids = input_ids[(cur_idxs_new[:,i] == 1)]
        cur_attention_mask = attention_mask[(cur_idxs_new[:,i] == 1)]
        

        cur_condition_feats = condition_feats[i].unsqueeze(0).expand(cur_input_ids.shape[0],-1,-1)
        total_len = len(cur_condition_feats)
        small_batch=25
        times = total_len//small_batch if total_len%small_batch==0 else total_len//small_batch+1

        for k in range(times):
            slice_input_ids = cur_input_ids[k*small_batch:(k+1)*small_batch]
            slice_attention_mask = cur_attention_mask[k*small_batch:(k+1)*small_batch]
            slice_condition_feats = cur_condition_feats[k*small_batch:(k+1)*small_batch]
            slice_scores = model.compute_slice_scores(slice_condition_feats, slice_input_ids, slice_attention_mask) 
            cur_scores.append(slice_scores)
        cur_scores = torch.cat(cur_scores,dim=0)

        cur_score_matrix_t_cond_new[:,i][(cur_idxs_new[:,i] == 1)] = cur_scores
        pbar.update(1)
    pbar.close()
    
    score_matrix_t_cond = ddp_allgather(cur_score_matrix_t_cond_new.T.contiguous()).T

    return score_matrix_t_cond

@torch.no_grad()
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

@torch.no_grad()
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



def remove_duplicates(sequence):
    '''
    Get the indices of the first occurrence of each element in the sequence.
    Used for deduplicating queries and/or gallery
    '''
    seen = set()
    out = []
    for i, x in enumerate(sequence):
        if x not in seen:
            seen.add(x)
            out.append(i)
    return np.array(out, dtype=np.int64)

def prepare_inputs(
    text_features, image_features, eeg_features,
    images, labels, ids, ids_txt, subjects,
    dedupe_gallery, dedupe_queries
):
    '''
    Prepare the features and additional data for retreval.
    Also apply deduplication of gallery and/or queries if specified.
    '''
    text_features = text_features.detach().cpu()
    image_features = image_features.detach().cpu()
    eeg_features = eeg_features.detach().cpu()
    
    num_images = image_features.shape[0]
    num_eegs = eeg_features.shape[0]

    labels_all = torch.cat(labels, dim=0).detach().cpu()

    if subjects is not None:
        subjects_all = list(subjects)
    else:
        subjects_all = None

    # indices before deduplication
    gallery_idx = np.arange(num_images)
    query_idx = np.arange(num_eegs)

    # Deduplicate gallery and/or queries 
    if dedupe_gallery:
        gallery_idx = remove_duplicates(ids)
    if dedupe_queries:
        query_idx = remove_duplicates(ids_txt)

    # Filter the features and labels based on the deduped indices
    eeg_queries = eeg_features[query_idx]
    img_gallery = image_features[gallery_idx]
    txt_gallery = text_features[gallery_idx]
    labels_gallery = labels_all[gallery_idx]
    gallery_ids = [ids[i] for i in gallery_idx] # list of image ids after deduping (we always dedupe gallery)
    query_ids = [ids_txt[i] for i in query_idx] # list of query (eeg) ids after deduping (if enabled)
    if subjects_all is not None:
        query_subjects = [subjects_all[i] for i in query_idx]
    else:
        query_subjects = None

    # Map iamge id -> label for queries (after deduping the indices have shifted, so we need to map the labels to the correct ids)
    # This is used later to look up the label for an id (useful for category-based eval)
    label_by_id = {} # this dict will have the image id as key and the label as value
    for i, gid in enumerate(gallery_ids):
        if gid not in label_by_id: # repeat writes should not happen after deduplication but just in case
            label_by_id[gid] = int(labels_gallery[i].item())
    
    # create a list where each entry is the label of the corresponding query id (or None if not found in gallery)
    # Again this is useful for category-based eval
    query_labels = [
        label_by_id[qid] if qid in label_by_id else None for qid in query_ids
    ]

    return {
        "eeg_queries": eeg_queries,
        "img_gallery": img_gallery,
        "txt_gallery": txt_gallery,
        "labels_gallery": labels_gallery,
        "gallery_ids": gallery_ids,
        "query_ids": query_ids,
        "query_labels": query_labels,
        "query_subjects": query_subjects,
        "query_idx": query_idx,
        "gallery_idx": gallery_idx,
        "num_queries": eeg_queries.shape[0],
        "num_images": img_gallery.shape[0]
    }

# Computes the actuial id-based retrieval metrics, i.e. only a match if the IDs are the same 
# Basically the same as copmute_metric_ret_area (with additional mrr, mean and median rank)
def compute_instance_metrics(ranked_indices, gallery_ids, query_ids):
    image_id_to_index = {k: v for v, k in enumerate(gallery_ids)} # dictionary of image id : index in gallery
    inst_ranks = [] #instance ranks
    inst_rrs = []   #reciprocal ranks
    top1 = top5 = top10 = 0
    valid = 0

    for i, query_id in enumerate(query_ids):
        target_image_index = image_id_to_index.get(query_id, None) # get the index of the target image in the gallery (using the id)
        row = ranked_indices[i].tolist() # row of the similarity matrix for this query, ranked from lowest to highest volume
        rank = row.index(target_image_index) + 1 # get the index of the target image in the row
        valid += 1
        inst_ranks.append(rank) # store the instance rank
        inst_rrs.append(1.0 / rank) # store the reciprocal rank
        if rank == 1: top1 += 1
        if rank <= 5: top5 += 1
        if rank <= 10: top10 += 1

    ranks_t = torch.tensor(inst_ranks, dtype=torch.float32) # convert to tensor for easy computation of mean and median

    return {
        "top1_accuracy": top1 / valid,
        "top5_accuracy": top5 / valid,
        "top10_accuracy": top10 / valid,
        "mean_rank": float(ranks_t.mean().item()),
        "median_rank": float(ranks_t.median().item()),
        "mrr": float(np.mean(inst_rrs)),
        "num_valid_queries": valid
    }


# This computes the label-based retrieval metrics, i.e. a correct label counts as a match
# Then it also does an additional light analysis of the results 
def category_and_analysis(
    ranked_indices, gallery_labels, query_labels,
    gallery_ids, query_ids, query_idx, query_subjects
):
    gallery_labels_np = gallery_labels.numpy()
    num_queries = ranked_indices.shape[0]

    category_ranks = []
    cat_rrs = [] # category reciprocal ranks
    category_aps = [] # category average precisions
    top1 = top5 = top10 = 0

    # dict  of img_id : idx, maps each img id to its position in the gallery. Used for quick lookup
    img_id_to_idx = {img_id: idx for idx, img_id in enumerate(gallery_ids)}
    
    # dict of label : accumulator, each accumulator is also a dict with the stats for that label
    per_label_accum = {}

    # list of dicts, each dict contains all info and metrics for a single query
    per_query_stats = []

    # loop over queries
    for query in range(num_queries):
        row = ranked_indices[query].tolist()
        query_label = query_labels[query]
        query_id = query_ids[query]

        # instance-based retrieval metrics again, used for later stats computation (yes I already computed this above, but oh well)
        inst_entry = {"instance_rank": None, "inst_hit@1": False, "inst_hit@5": False, "inst_hit@10": False} #initialize the stats
        gallery_pos = img_id_to_idx.get(query_id, None)
        inst_rank = row.index(gallery_pos) + 1
        inst_entry["instance_rank"] = inst_rank
        inst_entry["inst_hit@1"] = (inst_rank == 1)
        inst_entry["inst_hit@5"] = (inst_rank <= 5)
        inst_entry["inst_hit@10"] = (inst_rank <= 10)


        # Category-based retrieval metrics
        cat_entry = {"best_rank": None, "hit@1": False, "hit@5": False, "hit@10": False, "ap": 0.0} #initialize the stats

        # indices of all gallery items with the same label as the query
        relevant_positions = [j for j, label in enumerate(gallery_labels_np) if label == query_label]
        relevant_set = set(relevant_positions)
        # find best rank by iterating over the row until we find the first match
        for position, gallery_idx in enumerate(row):
            if gallery_idx in relevant_set:
                best_rank = position + 1
                cat_entry["best_rank"] = best_rank
                cat_entry["hit@1"] = (best_rank == 1)
                cat_entry["hit@5"] = (best_rank <= 5)
                cat_entry["hit@10"] = (best_rank <= 10)
                break

        # Average precision across all ranks
        # AP = mean of precision@rank over all ranks where a relevant item appears
        # precision@rank is the number of relevant items (up to current rank k) divided by k
        correct = 0
        ap_sum = 0.0
        for position, gallery_idx in enumerate(row, start=1): 
            if gallery_idx in relevant_set:
                correct += 1
                ap_sum += correct / float(position)
        cat_entry["ap"] = ap_sum / float(len(relevant_set))

        # Aggregate global category metrics for this query
        category_ranks.append(cat_entry["best_rank"])
        cat_rrs.append(1.0 / cat_entry["best_rank"])
        category_aps.append(cat_entry["ap"])
        if best_rank == 1: top1 += 1
        if best_rank <= 5: top5 += 1
        if best_rank <= 10: top10 += 1
        
        # Accumulate per-label metrics separately
        label = int(query_label)
        accumulator = per_label_accum.setdefault(label, {
                        "count": 0, "hits1": 0, "hits5": 0, "hits10": 0,
                        "ranks": [], "rr": [], "aps": []
                    })
        accumulator["count"] += 1
        accumulator["hits1"] += int(cat_entry["hit@1"])
        accumulator["hits5"] += int(cat_entry["hit@5"])
        accumulator["hits10"] += int(cat_entry["hit@10"])
        accumulator["ranks"].append(cat_entry["best_rank"])
        accumulator["rr"].append(1.0 / cat_entry["best_rank"])
        accumulator["aps"].append(cat_entry["ap"])
        
        # Combine info and metrics for this query into a single dict
        per_query_entry = {
            "local_index": query,
            "original_index": int(query_idx[query]),
            "id": query_id,
            "label": int(query_label),
            **inst_entry,
            **cat_entry
        }
        per_query_entry["subject"] = query_subjects[query]
        per_query_stats.append(per_query_entry)


    # compute category metrics for all queries
    if category_ranks:
        ranks_t = torch.tensor(category_ranks, dtype=torch.float32)
        category_metrics = {
            "top1_accuracy": top1 / len(category_ranks),
            "top5_accuracy": top5 / len(category_ranks),
            "top10_accuracy": top10 / len(category_ranks),
            "mean_rank": float(ranks_t.mean().item()),
            "median_rank": float(ranks_t.median().item()),
            "mrr": float(np.mean(cat_rrs)),
            "map": float(np.mean(category_aps)) ,
            "num_valid_queries": len(category_ranks)
        }
    else:
        category_metrics = {"error": "No valid category matches (no relevant labels in gallery)"} # should not happen

    # compute per-label metrics of all queries using the accumulators
    per_label_metrics = {}
    for label, accumulator in per_label_accum.items():
        count = max(1, accumulator["count"])
        ranks_t = torch.tensor(accumulator["ranks"], dtype=torch.float32) #easy mean and median computation
        per_label_metrics[int(label)] = {
            "num_queries": int(accumulator["count"]),
            "recall@1": accumulator["hits1"] / count,
            "recall@5": accumulator["hits5"] / count,
            "recall@10": accumulator["hits10"] / count,
            "mean_rank": float(ranks_t.mean().item()) if accumulator["ranks"] else None,
            "median_rank": float(ranks_t.median().item()) if accumulator["ranks"] else None,
            "mrr": float(np.mean(accumulator["rr"])) if accumulator["rr"] else 0.0,
            "map": float(np.mean(accumulator["aps"])) if accumulator["aps"] else 0.0,
        }

    # Order labels by best recall@1 to check which categories are easiest/hardest
    sorted_labels_by_r1 = sorted(
        per_label_metrics.items(), key=lambda kv: kv[1]["recall@1"], reverse=True
    )

    # Per subject metrics, both instance and category based
    per_subject_instance = {}
    per_subject_category = {}
    if query_subjects is not None:
        # create a dict of subject : list of query stats for that subject usning per_query_stats comptued above
        queries_by_subj = {}
        for elem in per_query_stats:
            queries_by_subj.setdefault(elem["subject"], []).append(elem)
        
        for subj, query_stats in queries_by_subj.items():
            # Instance metrics per subject
            inst_ranks_subj = [elem["instance_rank"] for elem in query_stats]
            if inst_ranks_subj:
                r_t = torch.tensor(inst_ranks_subj, dtype=torch.float32)
                per_subject_instance[subj] = {
                    "num_queries": len(inst_ranks_subj),
                    "recall@1": sum(e["inst_hit@1"] for e in query_stats) / len(inst_ranks_subj),
                    "recall@5": sum(e["inst_hit@5"] for e in query_stats) / len(inst_ranks_subj),
                    "recall@10": sum(e["inst_hit@10"] for e in query_stats) / len(inst_ranks_subj),
                    "mean_rank": float(r_t.mean().item()),
                    "median_rank": float(r_t.median().item()),
                    "mrr": float(np.mean([1.0 / r for r in inst_ranks_subj]))
                }

            # Category metrics per subject
            cat_ranks_subj = [entry["best_rank"] for entry in query_stats]
            if cat_ranks_subj:
                r_t = torch.tensor(cat_ranks_subj, dtype=torch.float32)
                per_subject_category[subj] = {
                    "num_queries": len(cat_ranks_subj),
                    "recall@1": sum(e["hit@1"] for e in query_stats) / len(cat_ranks_subj),
                    "recall@5": sum(e["hit@5"] for e in query_stats) / len(cat_ranks_subj),
                    "recall@10": sum(e["hit@10"] for e in query_stats) / len(cat_ranks_subj),
                    "mean_rank": float(r_t.mean().item()),
                    "median_rank": float(r_t.median().item()),
                    "mrr": float(np.mean([1.0 / r for r in cat_ranks_subj]))
                }

    # Group all the analysis data together
    analysis = {
        "per_label_category": per_label_metrics,
        "best_labels_by_recall@1": [int(l) for l, _ in sorted_labels_by_r1],
        "per_query": per_query_stats,
        "top_queries_by_category_rank": sorted(
            [q for q in per_query_stats if q["best_rank"] is not None],
            key=lambda q: q["best_rank"]
        ),
        "top_queries_by_instance_rank": sorted(
            [q for q in per_query_stats if q["instance_rank"] is not None],
            key=lambda q: q["instance_rank"]
        ), 
        "per_subject_instance": per_subject_instance if query_subjects is not None else None,
        "per_subject_category": per_subject_category if query_subjects is not None else None,
    }

    return category_metrics, analysis


# i am running out of names for these functions
@torch.no_grad()
def evaluate_retrieval(name, similarity_matrix, inputs):
    ranked_indices = torch.argsort(similarity_matrix, dim=-1, descending=False)
    instance = compute_instance_metrics(ranked_indices, inputs["gallery_ids"], inputs["query_ids"])
    category, analysis = category_and_analysis(
        ranked_indices,
        inputs["labels_gallery"],
        inputs["query_labels"],
        inputs["gallery_ids"],
        inputs["query_ids"],
        inputs["query_idx"],
        inputs["query_subjects"]
    )
    result = {
        "num_queries": inputs["num_queries"],
        "num_images": inputs["num_images"],
        "instance": instance,
        "category": category,
        "analysis": analysis
    }
    #Print the resutls
    print(f"\n=== {name} Retrieval ===")
    instance_metrics = instance
    print(f"[Instance] top1={instance_metrics['top1_accuracy']:.4f} top5={instance_metrics['top5_accuracy']:.4f} "
              f"top10={instance_metrics['top10_accuracy']:.4f} MR={instance_metrics['mean_rank']:.2f} MRR={instance_metrics['mrr']:.4f}")

    category_metrics = category
    atk_str = " ".join([f"map@{k}={category_metrics[f'{k}']:.4f}" for k in category_metrics.keys() if k.startswith("map@")])
    print(f"[Category] top1={category_metrics['top1_accuracy']:.4f} top5={category_metrics['top5_accuracy']:.4f} "
          f"top10={category_metrics['top10_accuracy']:.4f} MR={category_metrics['mean_rank']:.2f} "
          f"MRR={category_metrics['mrr']:.4f} MAP={category_metrics['map']:.4f} {atk_str}")
    
    return result


def compute_k_way_accuracy(similarity_matrix, gallery_ids, query_ids, k_values=(2, 4, 10, 50, 100, 200)):
    gallery_lookup = {gid: idx for idx, gid in enumerate(gallery_ids)}
    results = {k: [] for k in k_values}

    for q_idx, qid in enumerate(query_ids):
        target_idx = gallery_lookup.get(qid)
        if target_idx is None:
            continue
        for k in k_values:
            if k <= 1 or k > len(gallery_ids):
                continue
            distractors = np.random.default_rng(q_idx).choice(
                [i for i in range(len(gallery_ids)) if i != target_idx],
                size=k - 1,
                replace=False,
            )
            candidates = np.concatenate(([target_idx], distractors))
            sub_scores = similarity_matrix[q_idx, candidates]
            pred = candidates[sub_scores.argmin()]
            results[k].append(int(pred == target_idx))
    return {f"kway@{k}": np.mean(hits) if hits else 0.0 for k, hits in results.items()}

@torch.no_grad()
def compute_extended_retrieval_metrics(
    feat_t, feat_i, feat_e, images, labels,
    ids, ids_txt, subjects=None,
    dedupe_gallery=True,
    dedupe_queries=False,
    output_dir=None,
    file_prefix="retrieval",
    step=0
):
    inputs = prepare_inputs(
        feat_t, feat_i, feat_e,
        images, labels, ids, ids_txt, subjects,
        dedupe_gallery, dedupe_queries
    )

    # Similarity / volume matrices 
    vc3 = volume_computation3(inputs["eeg_queries"], inputs["img_gallery"], inputs["txt_gallery"])
    vc2_ev = volume_computation2(inputs["eeg_queries"], inputs["img_gallery"])
    vc2_et = volume_computation2(inputs["eeg_queries"], inputs["txt_gallery"])

    results = {}
    results["vc3"] = evaluate_retrieval("VC3 (EEG-Image-Text)", vc3, inputs)
    results["vc2_ev"] = evaluate_retrieval("VC2 (EEG-Image)", vc2_ev, inputs)
    results["vc2_et"] = evaluate_retrieval("VC2 (EEG-Text)", vc2_et, inputs)

    # store raw matrices and inputs for downstream viz/debug
    results["_raw"] = {
        "inputs": {
            **inputs,
            # convert numpy arrays for JSON later
            "gallery_idx": inputs["gallery_idx"].tolist() if hasattr(inputs["gallery_idx"], "tolist") else list(inputs["gallery_idx"]),
            "query_idx": inputs["query_idx"].tolist() if hasattr(inputs["query_idx"], "tolist") else list(inputs["query_idx"])
        }
    }

    #k-way accuracy
    results["k_way"] = compute_k_way_accuracy(vc3, inputs["gallery_ids"], inputs["query_ids"])
    LOGGER.info("[k-way] " + ", ".join(f"{k}={v:.4f}" for k, v in results["k_way"].items()))

    meta = {
        "dedupe_gallery": bool(dedupe_gallery),
        "dedupe_queries": bool(dedupe_queries)
    }
    results["meta"] = meta

    # Save results to JSON
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for key in ("vc3", "vc2_ev", "vc2_et", "k_way"):
            out_path = os.path.join(output_dir, f"{file_prefix}_{key}.json")
            with open(out_path, "w") as f:
                json.dump(results[key], f, indent=2)

    # Create retrieval visualization
    if output_dir:
        volumes = {"vc3": vc3, "vc2_ev": vc2_ev, "vc2_et": vc2_et}
        create_retrieval_viz(
            inputs=inputs,
            volumes=volumes,
            results=results,
            images=images,
            labels=labels,
            run_cfg=type("Cfg", (), {"output_dir": output_dir}),
            top_instances_n=5,
            step=step,
            save_prefix=file_prefix
        )

    return results



@torch.no_grad()
def create_retrieval_viz(
    inputs,
    volumes,              # dict with keys: "vc3", "vc2_ev", "vc2_et" (distance matrices)
    results=None,           #  metrics dict (for picking top instance-ranked queries)
    images=None,           
    labels=None,            
    run_cfg=None,       
    query_indices=None,     # explicit list of query indices (after deduping) to visualize
    top_instances_n=None,    # if set (e.g. 5) pick that many best instance-ranked queries (overrides query_indices)
    random_sample_n=5,      # fallback if neither query_indices nor top_instances_n provided
    step=0,
    save_prefix="retrieval"
):
    """
    Render retrieval thumbnails for three distance matrices (vc3, vc2_ev, vc2_et)
    using the EXACT ordering (dedup + alignment) contained in `inputs`.

    Column 0: the query's ground-truth image (if present in gallery).
    Columns 1..K: top-K gallery results (lower distance = better).
    Border colors:
      green  = exact instance match (same ID)
      orange = same label (category) but different instance
      red    = different label
    """

    output_dir = run_cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    imgs_to_show=10 # how many top results to show per query
    # 1. Unpack prepared retrieval data (already deduplicated / ordered)
    gallery_ids        = inputs["gallery_ids"]          # list of length G
    query_ids          = inputs["query_ids"]            # list of length Q
    gallery_labels_np  = inputs["labels_gallery"].cpu().numpy()
    original_gallery_idx = inputs["gallery_idx"]        # indices into original image tensor
    num_queries        = inputs["num_queries"]
    images = torch.cat(images, dim=0).cpu() if isinstance(images, (list, tuple)) else images

    # dict of gallery image id -> its index in the gallery (after deduping). used later for quick lookup
    gallery_id_to_idx = {gid: idx for idx, gid in enumerate(gallery_ids)}

    #  Decide which queries to visualize
    if top_instances_n is not None and results is not None: # pick top instance-ranked queries form resutls
        pick_mode = next((m for m in ("vc3","vc2_ev","vc2_et")), None)
        if pick_mode and "analysis" in results[pick_mode]:
            top_list = results[pick_mode]["analysis"].get("top_queries_by_instance_rank", [])[:top_instances_n]
            query_indices = [entry["local_index"] for entry in top_list]

    if query_indices is None:
        # Random 
        query_indices = np.random.choice(num_queries, size=random_sample_n, replace=False).tolist()

    # helper fnction to un-normalize the images to display
    mean = torch.tensor([0.5,0.5,0.5]).view(3,1,1)
    std  = torch.tensor([0.5,0.5,0.5]).view(3,1,1)
    def to_display(t):
        if t.dtype != torch.float32:
            t = t.float()
        img = (t * std + mean).clamp(0,1)
        return img.permute(1,2,0).numpy()

    # function to plot a single mode
    def plot_mode(mode_name, distance_matrix):

        ranked_gallery = torch.argsort(distance_matrix, dim=-1, descending=False)  # [Q, G]
        num_rows = len(query_indices)

        fig, axes = plt.subplots(num_rows, imgs_to_show + 1, figsize=(2.4*(imgs_to_show+1), 3*num_rows))

        # iterate over the indices of the queries to visualize
        for row_idx, q_idx in enumerate(query_indices):
            ax_query = axes[row_idx, 0] #first column is the query's ground-truth image
            ax_query.set_axis_off()

            query_id = query_ids[q_idx] # id of the query
            gallery_pos_of_query = gallery_id_to_idx.get(query_id, None) # position of the query's image in the gallery
            query_label = gallery_labels_np[gallery_pos_of_query]   # label of the query

            # Compute the instance rank
            instance_rank_text = "" # string to show what the instance rank for the query is (this is more useful if the queries are random)
            row_list = ranked_gallery[q_idx].tolist()   # sorted row of the matrix that corresponds to the query
            instance_rank = row_list.index(gallery_pos_of_query) + 1
            instance_rank_text = f"Best rank: {instance_rank}"
            
            # Show the query's ground truth image
            original_index = int(original_gallery_idx[gallery_pos_of_query])
            q_img = to_display(images[original_index])
            ax_query.imshow(q_img)
            ax_query.add_patch(patches.Rectangle(
                (0,0), q_img.shape[1], q_img.shape[0],
                linewidth=3, edgecolor='cyan', facecolor='none'
            ))

            ax_query.set_title(
                "\n".join(filter(None, [f"Query {q_idx}", f"Label {int(query_label)}",
                                        instance_rank_text])),
                fontsize=11
            )

            # do the same for the top 10 gallery images
            top_gallery_positions = ranked_gallery[q_idx][:imgs_to_show].tolist() # first 10 positions in the sorted row of the matrix
            for col, g_pos in enumerate(top_gallery_positions):
                ax_g = axes[row_idx, col + 1]
                ax_g.set_axis_off()

                g_id = gallery_ids[g_pos]
                g_label = int(gallery_labels_np[g_pos])
                is_instance = (g_id == query_id) #check if exact instance match
                is_same_label = (not is_instance) and (query_label is not None) and (g_label == query_label) # check if same label match

                border_color = 'green' if is_instance else ('orange' if is_same_label else 'red') #define border color

                if images is not None:
                    original_index = int(original_gallery_idx[g_pos])
                    if original_index < images.shape[0]:
                        g_img = to_display(images[original_index])
                        ax_g.imshow(g_img)
                        ax_g.add_patch(patches.Rectangle(
                            (0,0), g_img.shape[1], g_img.shape[0],
                            linewidth=3, edgecolor=border_color, facecolor='none'
                        ))

                dist_val = distance_matrix[q_idx, g_pos].item()
                ax_g.set_title(f"Rank{col+1}\nLabel {g_label}\n{dist_val:.4f}", fontsize=11, color='black')

        plt.suptitle(f"{mode_name} retrieval", fontsize=15)
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"{save_prefix}_{mode_name}_step{step}.png")
        plt.savefig(out_path, dpi=170)
        plt.close()
        print(f"[viz] saved {out_path}")


    # ------------------------------------------------------------------
    # 6. Render all three modes
    # ------------------------------------------------------------------
    if "vc3" in volumes:
        plot_mode("vc3", volumes["vc3"])
    if "vc2_ev" in volumes:
        plot_mode("vc2_ev", volumes["vc2_ev"])
    if "vc2_et" in volumes:
        plot_mode("vc2_et", volumes["vc2_et"])


def ATMS_accuracy(feat_t, feat_v, feat_e, labels, k):
    num_classes = 200
    all_classes = list(range(num_classes))

    correct = 0
    total = 0

    for idx, label in enumerate(labels):
        #label = label.item() if isinstance(label, torch.Tensor) else int(label)
        
        # k-way subset including the true class
        possible_classes = list(set(all_classes) - {label})
        selected_classes = random.sample(possible_classes, k - 1) + [label]

        # gather candidate class features
        v_batch = feat_v[selected_classes]          # [k, D]
        a_batch = feat_t[selected_classes]          # [k, D]
        e = feat_e[idx].unsqueeze(0)                # [1, D]

        # volume_computation3 returns [1, k]
        volumes = volume_computation3(e, v_batch, a_batch)[0]  # [k]

        pred_idx = torch.argmin(volumes).item()     # index in 0..k-1
        predicted_label = selected_classes[pred_idx]

        if predicted_label == label:
            correct += 1
        total += 1

    accuracy = correct / total
    print("Accuracy:", accuracy)
    return accuracy, labels#, features_tensor.cpu()
        