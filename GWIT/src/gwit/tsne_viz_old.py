import pandas as pd
import numpy as np
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import math
import random
from transformers import AutoTokenizer, PretrainedConfig, CLIPVisionModel
from torchvision import transforms
from utils import eval_collate_fn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
os.environ['HF_HOME'] = './Tesi/cache/'

### TODO Should use LSTM eeg encoder with pretrained weights at some point
@torch.no_grad()
def plot_tsne(model, args, global_step, accelerator, eval_dataset):
    # Filter out all but three most numerous labels
    labels_to_keep = [4, 3, 15] #4='espresso maker', 3='anemone fish', 15='cellular telephone'
    eval_dataset = eval_dataset.filter(lambda x: x['label'] in labels_to_keep) ## len = 228
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        shuffle=False,
        collate_fn=eval_collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers
    )
    eval_dataloader = accelerator.prepare(eval_dataloader)
    
    progress_bar = tqdm(
        range(0, math.ceil(len(eval_dataset)/args.train_batch_size)),
        initial=0,
        desc="Steps",
    )

    labels = []
    feat_t = []
    feat_i = []
    feat_e = []

    for step, batch in enumerate(eval_dataloader):
        feat_t.append(model.batch_get(batch, 'feat_t'))
        feat_i.append(model.batch_get(batch, 'feat_i'))
        feat_e.append(model.batch_get(batch, 'feat_e'))
        labels += batch["labels"]
        
        progress_bar.update(1)
        progress_bar.set_postfix({"step": step})

    feat_t = torch.cat(feat_t, dim=0).cpu().numpy()
    feat_i = torch.cat(feat_i, dim=0).cpu().numpy()
    feat_e = torch.cat(feat_e, dim=0).cpu().numpy()

    num_samples = feat_t.shape[0]

    # stack embeddings and build metadata
    X = np.vstack([feat_t, feat_i, feat_e])               # (3N, 512)
    modalities = (['text'] * num_samples) + (['image'] * num_samples) + (['eeg'] * num_samples)
    labels_all  = np.concatenate([labels, labels, labels])       # (3N,)

    # run t-SNE
    tsne = TSNE(n_components=2, perplexity=10, learning_rate='auto')
    fit_data = tsne.fit_transform(X)

    # make a DataFrame for seaborn
    df = pd.DataFrame({
        'x':         fit_data[:,0],
        'y':         fit_data[:,1],
        'modality':  modalities,
        'label':     labels_all
    })
    label_map = {3: 'anemone fish', 4: 'espresso maker', 15: 'cellular telephone'}
    df['label'] = df['label'].map(label_map)

    # plot
    plt.figure(figsize=(8,6))
    sns.scatterplot(
        data=df,
        x='x', y='y',
        hue='label',
        style='modality',
        markers={'text':'*', 'image':'s', 'eeg':'o'},
        palette='Set1',
        s=80
    )
    plt.title("t-SNE @ step {}".format(global_step))
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{args.output_dir}/tsne_plot_{global_step}.png")