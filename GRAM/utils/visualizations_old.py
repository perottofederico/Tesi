import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch
from tqdm import tqdm
import os
import json
import torch
import numpy as np
from time import time
import torch.distributed as dist
from tqdm import tqdm 
from torch.nn import functional as F
from utils.logger import LOGGER
from utils.distributed import  all_gather_list, ddp_allgather
from utils.tool import NoOp
from easydict import EasyDict as edict
from utils.volume import volume_computation4,volume_computation3, volume_computation5
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment as linear_assignment
import wandb
from dataset_EEG.name_map_ID import id_to_caption
import colorcet as cc
import glasbey
from matplotlib.patches import Ellipse
# Create legends on the right subplot
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
class TsnePlot:
    def __init__(self, perplexity=30, learning_rate='auto', n_iter=1000):
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def plot(self, embedding, labels, score, output_dir, step):
        # Perform t-SNE dimensionality reduction
        tsne = TSNE(perplexity=self.perplexity, learning_rate=self.learning_rate, max_iter=self.n_iter)
        reduced_embedding = tsne.fit_transform(embedding)

        max_val = np.max(reduced_embedding)
        min_val = np.min(reduced_embedding)
        reduced_embedding = (reduced_embedding - min_val)/(max_val - min_val)
        
        # Create scatter plot with different colors for different labels
        unique_labels = np.unique(labels)
        colors = list(cc.palette['glasbey_category10'])[:len(labels)]
        plt.figure(figsize=(15,12))
        fig, ax = plt.subplots()
        ax.tick_params(axis='both', labelsize=11)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(reduced_embedding[mask, 0], reduced_embedding[mask, 1], c=colors[i], label=f'{label}-{id_to_caption[label].split(",")[0]}', alpha=0.6)
        # why is it so hard to put the legend below the plot and not make it look weird damn
        #ax.legend(loc='lower right', fancybox=True, shadow=False, bbox_transform=fig.transFigure, ncol=4, bbox_to_anchor=(1, -0.2))
        ax.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0, ncol=2)
        #plt.tight_layout()
        plt.savefig('{}/tsne_eeg_kmean_{:.5f}_step_{}.png'.format(output_dir, score, step), bbox_inches="tight", dpi=300)
        print("saved tsne plot to {}/tsne_eeg_kmean_{}.png".format(output_dir, score))
        plt.close()
        return reduced_embedding
    
class K_means:
    def __init__(self, n_clusters=40, random_state=45):
        self.n_clusters = n_clusters
        self.random_state = random_state
        
    def transform(self, embed, gt_labels):
        pred_labels = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit_predict(embed)
        score       = self.cluster_acc(gt_labels, pred_labels)
        # image_score = K_means_model.score(image_embed, KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit_predict(image_embed))
        return score

    # Thanks to: https://github.com/k-han/DTC/blob/master/utils/util.py
    def cluster_acc(self, y_true, y_pred):
        """
        Calculate clustering accuracy. Require scikit-learn installed
        # Arguments
            y: true labels, numpy.array with shape `(n_samples,)`
            y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        # Return
            accuracy, in [0,1]
        """
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / y_pred.size
    


    # used gathered embeddings to create plots


@torch.no_grad()
def create_plots(model, val_loader, args, tasks):

    # additional data for t-SNE
    subjects_list = []
    labels_list = []
    feat_t = []
    feat_e = []
    feat_v = []
 
    main_task = tasks.split('%')[0]
    subtasks = tasks.split('%')[1:]
    store_dict = {}

    # these are just for the tsne plot of eeg embeddings
    eeg_featvec_proj  = np.array([])
    labels_array      = np.array([])

    for task in subtasks:
        store_dict[f'condition_feats_{task}'] = []        

    for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
        batch = edict(batch)
        evaluation_dict= model(batch, tasks.split('%')[0], compute_loss=False)

        feat_t.append(evaluation_dict['feat_t'])
        feat_v.append(evaluation_dict['feat_v'])
        feat_e.append(evaluation_dict['feat_e'])

        eeg_featvec_proj = np.concatenate((eeg_featvec_proj, evaluation_dict['feat_e'].cpu().detach().numpy()), axis=0) if eeg_featvec_proj.size else evaluation_dict['feat_e'].cpu().detach().numpy()
      
        # Additional data for tsne plot
        if 'eeg_subjects' in batch:
            subjects_list.extend(batch['eeg_subjects'].cpu().numpy().tolist())
        #if 'raw_captions' in batch:
        #    captions_list.extend(batch['raw_captions'])  # Limit for memory
        if 'labels' in batch:
            labels_list.extend(batch['labels'].cpu().numpy().tolist())
            labels_array = np.concatenate((labels_array, batch['labels'].cpu().detach().numpy()), axis=0) if labels_array.size else batch['labels'].cpu().detach().numpy()
  

    feat_t = torch.cat(feat_t, dim = 0)
    feat_t = ddp_allgather(feat_t)

    feat_e = torch.cat(feat_e, dim = 0)
    feat_e = ddp_allgather(feat_e)

    feat_v = torch.cat(feat_v, dim = 0)
    feat_v = ddp_allgather(feat_v)

    # gather step for the additional data
    subjects_list = [j for i in all_gather_list(subjects_list) for j in i]
    #captions_list = [j for i in all_gather_list(captions_list) for j in i]
    labels_list = [j for i in all_gather_list(labels_list) for j in i]
    
    if dist.get_rank() == 0:
        step = args.run_cfg.checkpoint.split('_')[-1].replace('.pt','')

        # generate tsne plot
        tsne_data = {
            'text': feat_t.cpu().numpy(),
            'vision': feat_v.cpu().numpy(), 
            'eeg': feat_e.cpu().numpy(),
            'subjects': np.array(subjects_list[:len(feat_t)]),
            #'captions': captions_list[:len(feat_t)]
            'labels': labels_list[:len(feat_t)]
        }
        
        eeg_tsne_data = {
            'eeg': eeg_featvec_proj,
            'labels': labels_array
        }
        
        plot_tsne_eeg_embeddings(eeg_tsne_data, args.run_cfg.output_dir, step)
        plot_tsne_all_modalities(tsne_data, args.run_cfg.output_dir, step) 
        plot_tsne_pairwise_modalities(tsne_data, args.run_cfg.output_dir, step)
        plot_alignment_heatmap(tsne_data, args.run_cfg.output_dir, step)
        plot_volume_heatmap_per_sample(tsne_data, args.run_cfg.output_dir, step)
        out = plot_tsne_centroids(
            feat_t, feat_v, feat_e, labels_array,
            output_dir=f'{args.run_cfg.output_dir}',
            title=f'Centroids - step {step}',
            step=step
        )

# tsne viz of only the eeg embeddings, colored by class, with k-means score
@torch.no_grad()
def plot_tsne_eeg_embeddings(data, output_dir, step):

    k_means        = K_means(n_clusters=40)
    clustering_acc_proj = k_means.transform(data["eeg"], data["labels"])
    print("[Test KMeans score Proj: {}]".format(clustering_acc_proj))
    
    tsne_plot = TsnePlot(perplexity=30, learning_rate='auto', n_iter=1000)
    tsne_plot.plot(data["eeg"], data["labels"], clustering_acc_proj, output_dir, step)

# tsne plot of all modalities (eeg, image, text) + tsne plot with lines
@torch.no_grad()
def plot_tsne_all_modalities(data, output_dir, step):

    # Define a list of labels to visualize to maybe reduce clutter
    #labels_to_keep = [0, 4, 16, 37]
    #label_names = {0: 'sorrel', 4: 'espresso maker', 16: 'golf ball', 37: 'locomotive'}
    #label_colors = {0: '#ff6d60', 4: '#19b99c', 16: '#bf00ff', 37: "#eccf0d"}
    #unique_labels = sorted(set(labels_to_keep))
    
    labels_to_keep = data['labels']  # Use all labels in the dataset
    label_names = {label: f'{label}' for label in labels_to_keep}
    unique_labels = sorted(set(labels_to_keep))

    label_colors = list(cc.palette['glasbey_category10'])[:40]


    embedding_marker = {'Text': '*', 'Image': 's', 'EEG': '^'}
    labels_list = data['labels']
    labels_array = np.array(labels_list)
    labels_to_keep_mask = np.isin(labels_array, labels_to_keep)

    # Concatenate all embeddings
    feat_t = data['text'][labels_to_keep_mask]
    feat_i = data['vision'][labels_to_keep_mask]
    feat_e = data['eeg'][labels_to_keep_mask]
    labels_list = labels_array[labels_to_keep_mask].tolist()
    
    print(f"Collected embeddings: Text={feat_t.shape}, Image={feat_i.shape}, EEG={feat_e.shape}")
    
    # Create combined embedding matrix
    #all_embeddings = np.concatenate([feat_t, feat_i, feat_e], axis=0)
    all_embeddings = np.vstack([feat_t, feat_i, feat_e])

    # Create modality labels to distinguish the modality types in the plot
    n_samples = feat_t.shape[0]
    modality_labels = ['Text'] * n_samples + ['Image'] * n_samples + ['EEG'] * n_samples
    class_labels = labels_list * 3  # Repeat labels for each modality

    print(f"Running t-SNE on {all_embeddings.shape[0]} embeddings...")
    
    # Run t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        max_iter=1000,
        learning_rate='auto',
        init='pca'
    )
    
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # PLot the points
    fig, (ax_plot, ax_legend) = plt.subplots(1, 2, figsize=(16, 8), 
                                             gridspec_kw={'width_ratios': [3, 1]})
    

    # Plot on the left subplot
    for modality in ['Text', 'Image', 'EEG']: 
        for label in unique_labels: #for each label
            modality_mask = np.array(modality_labels) == modality # create a mask to filter the points to the current modality (eeg/image/text)
            class_mask = np.array(class_labels) == label # create a mask to filter the points to the current class/label
            mask = modality_mask & class_mask # combine the two masks 
            
            # plot the points that belong to the current modality and class
            ax_plot.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                          marker=embedding_marker[modality],
                          c=[label_colors[label]], 
                          alpha=0.7, s=50)

    ax_plot.set_title(f'Multimodal Embedding Space (Step {step})')
    ax_plot.set_xlabel('t-SNE Dimension 1')
    ax_plot.set_ylabel('t-SNE Dimension 2')
    ax_plot.grid(True, alpha=0.3)
    

    # plot legends
    # There's actually two legends, one for the modalities (eeg/image/text) and one for the labels
    # Hide the right subplot axes
    ax_legend.axis('off')
    
    # create elements for the modality legend, each with its own marker
    modality_elements = [Line2D([0], [0], marker=marker, color='black', linestyle='None', markersize=10, label=modality)
                            for modality, marker in embedding_marker.items()]
    
    # create elements for the label legend, each with its own color
    label_elements = []
    for label in unique_labels:
        from dataset_EEG.name_map_ID import id_to_caption
        label_name = id_to_caption[label].split(',')[0] # take the label and ignore anything after the first comma(if there is one)
        label_elements.append(Rectangle((0, 0), 1, 1, facecolor=label_colors[label], 
                                      label=f'{label}-{label_name}'))
    
    # Add legends to the right subplot
    leg1 = ax_legend.legend(handles=modality_elements, title="Modalities", 
                           loc='upper left', fontsize=11)
    ax_legend.add_artist(leg1)
    
    leg2 = ax_legend.legend(handles=label_elements, title="Labels", 
                           loc='center left', fontsize=11,
                           ncol=1 if len(unique_labels) <= 20 else 2) 
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/tsne_all_modalities_step_{step}.png', dpi=300, bbox_inches='tight')
    plt.close()
    

    # Try to visualize alignment by connecting matching samples
    fig, (ax_plot, ax_legend) = plt.subplots(1, 2, figsize=(16, 8), 
                                             gridspec_kw={'width_ratios': [3, 1]})
    
    #same plotting code as above
    for modality in ['Text', 'Image', 'EEG']:
        for label in labels_to_keep:
            modality_mask = np.array(modality_labels) == modality
            class_mask = np.array(class_labels) == label
            mask = modality_mask & class_mask
            
            label_name = label_names.get(label, f'Label {label}')
            ax_plot.scatter(
                embeddings_2d[mask, 0], 
                embeddings_2d[mask, 1],
                marker = embedding_marker[modality],
                c = label_colors[label],
                alpha=0.7,
                s=50
            )

    ax_plot.set_title(f'Multimodal Alignment Embedding Space (Step {step})')
    ax_plot.set_xlabel('t-SNE Dimension 1')
    ax_plot.set_ylabel('t-SNE Dimension 2')
    ax_plot.grid(True, alpha=0.3)
    
    # Draw lines connecting matching samples (sample every nth to avoid clutter)
    connection_indices = [] # list of indices of points to connect
    for label in unique_labels:
        # Find all indices for the current label and take the first one
        label_mask = np.array(labels_list) == label
        label_indices = np.where(label_mask)[0]
        connection_indices.append(label_indices[0])
        #connection_indices.append(np.random.choice(label_indices))

    connection_indices = np.array(connection_indices)
    
    #plot the lines
    for i in connection_indices:
        text_point = embeddings_2d[i]
        image_point = embeddings_2d[i + n_samples]
        eeg_point = embeddings_2d[i + 2*n_samples]
        
        #ax_plot.plot([text_point[0], image_point[0]], [text_point[1], image_point[1]], 
        #        'k-', alpha=0.4, linewidth=0.7)
        ax_plot.plot([text_point[0], eeg_point[0]], [text_point[1], eeg_point[1]], 
                'k-', alpha=0.4, linewidth=0.7)
        ax_plot.plot([image_point[0], eeg_point[0]], [image_point[1], eeg_point[1]], 
                'k-', alpha=0.4, linewidth=0.7)
    
     # Add legends to the right subplot as above
    ax_legend.axis('off')
    leg1 = ax_legend.legend(handles=modality_elements, title="Modalities", 
                           loc='upper left', fontsize=11)
    ax_legend.add_artist(leg1)
    
    leg2 = ax_legend.legend(handles=label_elements, title="Labels", 
                           loc='center left', fontsize=11,
                           ncol=1 if len(unique_labels) <= 20 else 2)
    
    plt.tight_layout()  # leave room on the right for the legend
    plt.savefig(f'{output_dir}/tsne_all_modalities_alignment_step_{step}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    ## 3. Subject-based visualization (if subject info available)
    #if subject_ids > 1:
    #    plt.figure(figsize=(12, 8))
    #    
    #    unique_subjects = sorted(list(set(subject_ids)))
    #    subject_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_subjects)))
    #    
    #    for subplot_idx, modality in enumerate(['Text', 'Image', 'EEG']):
    #        plt.subplot(1, 3, subplot_idx + 1)
    #        
    #        modality_mask = np.array(modality_labels) == modality
    #        modality_embeddings = embeddings_2d[modality_mask]
    #        modality_subjects = subject_ids  # Same subjects for all modalities
    #        
    #        for i, subject in enumerate(unique_subjects):
    #            subject_mask = np.array(modality_subjects) == subject
    #            if np.any(subject_mask):
    #                plt.scatter(
    #                    modality_embeddings[subject_mask, 0],
    #                    modality_embeddings[subject_mask, 1],
    #                    c=[subject_colors[i]], 
    #                    label=f'Subject {subject}', 
    #                    alpha=1.0,
    #                    s=50
    #                )
    #        
    #        plt.title(f'{modality} Embeddings by Subject')
    #        plt.xlabel('t-SNE Dimension 1')
    #        plt.ylabel('t-SNE Dimension 2')
    #        if subplot_idx == 2:  # Only show legend on last subplot
    #            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #        plt.grid(True, alpha=0.3)
    #    
    #    plt.tight_layout()
    #    plt.savefig(f'{output_dir}/tsne_subjects_step_{step}.png', dpi=300, bbox_inches='tight')
    #    plt.close()
    #
    print(f"t-SNE visualizations saved to {output_dir}/{step}")

# plot tsne viz of text-image, eeg-image and text-eeg 
@torch.no_grad()
def plot_tsne_pairwise_modalities(data, output_dir, step=0):
    
    feat_t, feat_i, feat_e = data['text'], data['vision'], data['eeg']
    labels = np.array(data['labels'])
    
    # Plot with connections
    unique_labels = np.unique(labels)
    colors = list(cc.palette['glasbey_category10'])[:40]


    #plot title, name suffix, feat1, feat2, marker1, marker2
    pairs = [
        ("Text-Image", "text_image", feat_t, feat_i, 'o', 's'),
        ("Text-EEG", "text_eeg", feat_t, feat_e, 'o', '^'), 
        ("Image-EEG", "image_eeg", feat_i, feat_e, "s", '^')
    ]
    
    for title, suffix, feat1, feat2, m1, m2 in pairs:
        # Combine and run t-SNE on pair
        combined = np.vstack([feat1, feat2]) 
        tsne = TSNE(n_components=2,
                        perplexity=30,
                        random_state=42,
                        max_iter=1000,
                        learning_rate='auto',
                        init='pca'
                        )
        
        embedded = tsne.fit_transform(combined) 
        
        n_samples = len(feat1) 

        fig, (ax_plot, ax_legend) = plt.subplots(1,2,figsize=(16, 8),
                                                gridspec_kw={'width_ratios': [3, 1]})
        for i, label in enumerate(unique_labels):
            mask = labels == label  # Shape: (333,) - boolean mask
            # Get indices where mask is True
            true_indices = np.where(mask)[0]  # Get actual indices, not boolean mask
            
            # First modality 
            first_modality_indices = true_indices 
            ax_plot.scatter(embedded[first_modality_indices, 0], embedded[first_modality_indices, 1], 
                            c=[colors[i]], marker=m1, alpha=0.7, s=50, 
            )#label=f'{title.split("-")[0]} - {(id_to_caption[label]).split(",")[0]}')
            
            # Second modality 
            second_modality_indices = true_indices + n_samples  # Shift by n_samples for second modality
            ax_plot.scatter(embedded[second_modality_indices, 0], embedded[second_modality_indices, 1],
                            c=[colors[i]], marker=m2, alpha=0.7, s=50,
            )#label=f'{title.split("-")[1]} - {(id_to_caption[label]).split(",")[0]}')
        
        ax_plot.set_title(f'{title} Alignment')
        ax_plot.grid(True, alpha=0.3)
        #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)
    
        ax_legend.axis('off')

        # create elements for the modality legend, each with its own marker
        embedding_marker = {title.split('-')[0]: m1, title.split('-')[1]: m2}
        modality_elements = [Line2D([0], [0], marker=marker, color='black', 
                                   linestyle='None', markersize=10, label=modality)
                            for modality, marker in embedding_marker.items()]

        # create elements for the label legend, each with its own color
        label_elements = []
        for label in unique_labels:
            from dataset_EEG.name_map_ID import id_to_caption
            label_name = id_to_caption[label].split(',')[0]
            label_elements.append(Rectangle((0, 0), 1, 1, facecolor=colors[label], 
                                          label=f'{label}-{label_name}'))

        # Add legends to the right subplot
        leg1 = ax_legend.legend(handles=modality_elements, title="Modalities", 
                               loc='upper left', fontsize=11)
        ax_legend.add_artist(leg1)

        leg2 = ax_legend.legend(handles=label_elements, title="Labels", 
                               loc='center left', fontsize=11,
                               ncol=1 if len(unique_labels) <= 20 else 2) # use 2 columns if too many labels


        plt.tight_layout()
        plt.savefig(f'{output_dir}/tsne_pairwise_{suffix}_step_{step}.png', dpi=300, bbox_inches='tight')
        plt.close()

# plot tsne viz of all and pairwise modalities, but using centroids
@torch.no_grad()
def plot_tsne_centroids(
    text_feats, image_feats, eeg_feats, labels,
    output_dir='centroids.png',
    random_state=42,
    title='Multimodal Latent Space',
    annotate_centroids=True,
    step = 0
):
    """
    plots the latent space with tsne of of all 3 modalities, as well as only eeg-image
    Can do either all the points, a subset of the points or centroids for each class and modality 
    (and optionally che covariance ellipses but it looks terrible)
    """

    device = text_feats.device
    labels = torch.as_tensor(labels).long().cpu()
    labels_array = np.array(labels)
    unique_labels = np.unique(labels_array)

    T = text_feats.cpu()
    V = image_feats.cpu()
    E = eeg_feats.cpu()

    C = int(labels.max().item()) + 1
    #colors = generate_distinct_colors(C)
    #colors = glasbey.create_palette(palette_size=40)
    colors = list(cc.palette['glasbey_category10'])[:40]
    

    def compute_centroids(feats, labels, C):
        cents = []
        for c in range(C):
            mask = (labels == c)
            cents.append(feats[mask].mean(dim=0, keepdim=True))
        return torch.cat(cents, dim=0)

    centroid_t = compute_centroids(T, labels, C)
    centroid_v = compute_centroids(V, labels, C)
    centroid_e = compute_centroids(E, labels, C)


    # Combine the centroids for all modalities viz
    all_centroids = torch.cat([centroid_t, centroid_v, centroid_e], dim=0).numpy()
    modality_labels = (['Text'] * C) + (['Image'] * C) + (['EEG'] * C)
    class_labels = list(range(C)) * 3 # repeat each class per modality
    n_points = all_centroids.shape[0]

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate='auto',
        init='pca',
        random_state=random_state,
        max_iter=1000
    )
    emb2d = tsne.fit_transform(all_centroids)


    fig, (ax_plot, ax_legend) = plt.subplots(1, 2, figsize=(16, 8),
                                             gridspec_kw={'width_ratios': [3, 1]})
    ax_legend.axis('off')
    modality_markers = {'Text': 'o', 'Image': 's', 'EEG': '^'}

    # Scatter centroids
    for modality in ['Text', 'Image', 'EEG']:
        m_mask = np.array(modality_labels) == modality
        for c in range(C):
            mc_mask = (np.array(class_labels) == c) & m_mask
            if not np.any(mc_mask):
                continue
            ax_plot.scatter(
                emb2d[mc_mask, 0],
                emb2d[mc_mask, 1],
                marker=modality_markers[modality],
                c=[colors[c]],
                edgecolors='k',
                linewidths=0.7,
                s=260,
                alpha=0.95,
                zorder=5
            )
            # Annotate only once per class (on Text centroids) if requested
            if annotate_centroids and modality == 'Text':
                ax_plot.text(
                    emb2d[mc_mask, 0],
                    emb2d[mc_mask, 1],
                    str(c),
                    ha='center',
                    va='center',
                    fontsize=11,
                    color='k',
                    zorder=6
                )
    ax_plot.set_title(f'{title} (Centroids Only)')
    ax_plot.set_xlabel('Dim 1')
    ax_plot.set_ylabel('Dim 2')
    ax_plot.grid(True, alpha=0.3)


    # Legends
    modality_elements = [
        Line2D([0], [0], marker=m, color='black', linestyle='None',
               markersize=10, label=mod)
        for mod, m in modality_markers.items()
    ]
    label_elements = []
    for c in range(C):
        from dataset_EEG.name_map_ID import id_to_caption
        label_name = id_to_caption.get(c, str(c)).split(',')[0]
        label_elements.append(
            Rectangle((0, 0), 1, 1, facecolor=colors[c], label=f'{c}-{label_name}')
        )

    leg1 = ax_legend.legend(handles=modality_elements, title="Modalities",
                            loc='upper left', fontsize=11)
    ax_legend.add_artist(leg1)
    leg2 = ax_legend.legend(handles=label_elements, title="Labels",
                            loc='center left', fontsize=11,
                            ncol=1 if C <= 20 else 2)

    fig.tight_layout()
    fig.savefig(f'{output_dir}/tsne_all_modalities_centroids', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # now do the pairwise plots too
    pairs = [
        ("Text-Image", "text_image", centroid_t, centroid_v, 'o', 's'),
        ("Text-EEG", "text_eeg", centroid_t, centroid_e, 'o', '^'), 
        ("Image-EEG", "image_eeg", centroid_v, centroid_e, "s", '^')
    ]
    for title, suffix, feat1, feat2, m1, m2 in pairs:
        # Combine and run t-SNE on pair
        combined = np.vstack([feat1, feat2]) 
        tsne = TSNE(
                    n_components=2,
                    perplexity=30,
                    learning_rate='auto',
                    init='pca',
                    random_state=random_state,
                    max_iter=1000
                )
        embedded = tsne.fit_transform(combined) 
        n_samples = len(feat1)

        fig, (ax_plot, ax_legend) = plt.subplots(1,2,figsize=(16, 8),
                                                gridspec_kw={'width_ratios': [3, 1]})
        for c in range(C):
            ax_plot.scatter(
                embedded[c, 0], embedded[c, 1],
                marker=m1,
                c=[colors[c]],
                edgecolors='k',
                linewidths=0.6,
                s=260
            )
            ax_plot.scatter(
                embedded[C + c, 0], embedded[C + c, 1],
                marker=m2,
                c=[colors[c]],
                edgecolors='k',
                linewidths=0.6,
                s=260
            )
            ax_plot.text(
                embedded[c, 0], embedded[c, 1], str(c),
                ha='center', va='center', fontsize=11, color='k'
            )

        ax_plot.set_title(f'{title} Centroids')
        ax_plot.grid(True, alpha=0.3)

        ax_legend.axis('off')

        embedding_marker = {title.split('-')[0]: m1, title.split('-')[1]: m2}
        modality_elements = [Line2D([0], [0], marker=marker, color='black', 
                                   linestyle='None', markersize=10, label=modality)
                            for modality, marker in embedding_marker.items()]

        label_elements = []
        for label in range(C):
            from dataset_EEG.name_map_ID import id_to_caption
            label_name = id_to_caption[label].split(',')[0]
            label_elements.append(Rectangle((0, 0), 1, 1, facecolor=colors[label], 
                                          label=f'{label}-{label_name}'))

        # Add legends to the right subplot
        leg1 = ax_legend.legend(handles=modality_elements, title="Modalities", 
                               loc='upper left', fontsize=11)
        ax_legend.add_artist(leg1)

        leg2 = ax_legend.legend(handles=label_elements, title="Labels", 
                               loc='center left', fontsize=11,
                               ncol=1 if C <= 20 else 2) # use 2 columns if too many labels


        plt.tight_layout()
        plt.savefig(f'{output_dir}/tsne_centroids_pairwise_{suffix}_step_{step}.png', dpi=300, bbox_inches='tight')
        print(f"{output_dir}/tsne_centroids_pairwise_{suffix}_step_{step}.png")
        plt.close()
    

    return output_dir

@torch.no_grad()
def plot_alignment_heatmap(data, output_dir, step=0):
    """Create alignment quality heatmap per class."""
    feat_t, feat_i, feat_e = data['text'], data['vision'], data['eeg']
    labels = np.array(data['labels'])
    
    feat_t_tensor = torch.tensor(feat_t, dtype=torch.float16)
    feat_i_tensor = torch.tensor(feat_i, dtype=torch.float16)
    feat_e_tensor = torch.tensor(feat_e, dtype=torch.float16)

    # Normalize features
    #from sklearn.preprocessing import StandardScaler
    #feat_t = StandardScaler().fit_transform(feat_t)
    #feat_i = StandardScaler().fit_transform(feat_i)
    #feat_e = StandardScaler().fit_transform(feat_e)
    
    unique_labels = np.unique(labels)
    print("number of unique labels:", len(unique_labels))
    unique, counts = np.unique(labels, return_counts=True)
    print("Unique labels count:", dict(zip(unique, counts)))
    alignment_matrix = np.zeros((len(unique_labels), 4))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if np.sum(mask) > 1:
            # Compute mean cosine similarity for this class
            t_class = feat_t[mask]
            i_class = feat_i[mask] 
            e_class = feat_e[mask]
            
            # Text-Image alignment
            alignment_matrix[i, 0] = np.mean([
                np.dot(t_class[j], i_class[j]) / (np.linalg.norm(t_class[j]) * np.linalg.norm(i_class[j]))
                for j in range(len(t_class))
            ])
            
            # Text-EEG alignment  
            alignment_matrix[i, 1] = np.mean([
                np.dot(t_class[j], e_class[j]) / (np.linalg.norm(t_class[j]) * np.linalg.norm(e_class[j]))
                for j in range(len(t_class))
            ])
            
            # Image-EEG alignment
            alignment_matrix[i, 2] = np.mean([
                np.dot(i_class[j], e_class[j]) / (np.linalg.norm(i_class[j]) * np.linalg.norm(e_class[j]))
                for j in range(len(i_class))
            ])

            # Overall gramian volume of the triad of embeddings
            t_class_tensor = feat_t_tensor[mask]
            i_class_tensor = feat_i_tensor[mask]
            e_class_tensor = feat_e_tensor[mask]
            volumes = []
            for j in range(len(t_class)):
                volume = volume_computation3(
                    t_class_tensor[j:j+1],  # Single sample as tensor
                    i_class_tensor[j:j+1], 
                    e_class_tensor[j:j+1]
                )
                volumes.append(volume.item() if isinstance(volume, torch.Tensor) else volume)
            alignment_matrix[i, 3] = np.mean(volumes)

    plt.figure(figsize=(10, max(6, len(unique_labels) * 0.3)))
    sns.heatmap(alignment_matrix, 
                xticklabels=['Text-Image', 'Text-EEG', 'Image-EEG', 'Volume'],
                yticklabels=[f'{(id_to_caption[label]).split(",")[0]}' for label in unique_labels],
                annot=True, fmt='.3f', cmap='viridis', center=0)
    plt.title(f'Cross-Modal Alignment by Class (Step {step})')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/alignment_heatmap_step_{step}.png', dpi=300, bbox_inches='tight')
    plt.close()

@torch.no_grad()
def plot_volume_heatmap_per_sample(data, output_dir, step=0):
    """Create heatmap showing volume for each individual embedding triad, organized by label."""
    feat_t, feat_i, feat_e = data['text'], data['vision'], data['eeg']
    labels = np.array(data['labels'])
    
    # Convert to tensors for volume computation
    feat_t_tensor = torch.tensor(feat_t, dtype=torch.float32)
    feat_i_tensor = torch.tensor(feat_i, dtype=torch.float32)
    feat_e_tensor = torch.tensor(feat_e, dtype=torch.float32)
    
    unique_labels = np.unique(labels)
    print(f"Computing volumes for {len(unique_labels)} labels...")
    
    # Compute volumes for each sample
    all_volumes = []
    for i in range(len(labels)):
        volume = volume_computation3(
            feat_t_tensor[i:i+1],
            feat_i_tensor[i:i+1], 
            feat_e_tensor[i:i+1]
        )
        all_volumes.append(volume.item() if isinstance(volume, torch.Tensor) else volume)
    
    # Organize volumes by label
    volumes_by_label = {}
    max_samples_per_label = 0
    
    for label in unique_labels:
        mask = labels == label
        label_volumes = np.array(all_volumes)[mask]
        volumes_by_label[label] = label_volumes
        max_samples_per_label = max(max_samples_per_label, len(label_volumes))
    
    print(f"Max samples per label: {max_samples_per_label}")
    
    # Create matrix with NaN for missing values
    volume_matrix = np.full((len(unique_labels), max_samples_per_label), np.nan)
    
    for i, label in enumerate(unique_labels):
        label_volumes = volumes_by_label[label]
        volume_matrix[i, :len(label_volumes)] = label_volumes
    
    # Create the heatmap
    plt.figure(figsize=(max(12, max_samples_per_label * 0.5), max(8, len(unique_labels) * 0.4)))
    
    # Use a mask to hide NaN values
    mask = np.isnan(volume_matrix)
    
    # Create heatmap with custom colormap
    sns.heatmap(volume_matrix, 
                mask=mask,
                yticklabels=[f'{(id_to_caption[label]).split(",")[0]} (n={len(volumes_by_label[label])})' for label in unique_labels],
                xticklabels=[f'Sample {i+1}' for i in range(max_samples_per_label)],
                annot=True, fmt='.3f',  # cahnge to false to remove number annotations
                cmap='viridis',
                cbar_kws={'label': 'Gramian Volume'})
    
    plt.title(f'Volume Spanned by Embedding Triads per Sample (Step {step})')
    plt.xlabel('Sample Index within Label')
    plt.ylabel('Label (Class)')
    
    # Rotate x-axis labels if there are many samples
    if max_samples_per_label > 20:
        plt.xticks(rotation=90)
        # Show only every nth x-tick to avoid clutter
        n_ticks = min(20, max_samples_per_label)
        tick_indices = np.linspace(0, max_samples_per_label-1, n_ticks, dtype=int)
        plt.xticks(tick_indices, [f'Sample {i+1}' for i in tick_indices])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/volume_heatmap_per_sample_step_{step}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a summary statistics plot
    plt.figure(figsize=(12, 8))
    
    # Box plot showing distribution of volumes per label
    volume_data = []
    volume_labels = []
    
    for label in unique_labels:
        label_volumes = volumes_by_label[label]
        volume_data.extend(label_volumes)
        volume_labels.extend([f'{(id_to_caption[label]).split(",")[0]}'] * len(label_volumes))
    
    # Create DataFrame for easier plotting
    import pandas as pd
    df = pd.DataFrame({'Volume': volume_data, 'Label': volume_labels})
    
    # Box plot
    plt.subplot(2, 1, 1)
    sns.boxplot(data=df, x='Label', y='Volume')
    plt.title(f'Distribution of Volumes by Label (Step {step})')
    plt.xticks(rotation=90)
    plt.grid(True, alpha=0.3)
    
    # Bar plot of mean volumes
    plt.subplot(2, 1, 2)
    mean_volumes = [np.mean(volumes_by_label[label]) for label in unique_labels]
    std_volumes = [np.std(volumes_by_label[label]) for label in unique_labels]
    
    bars = plt.bar([f'{(id_to_caption[label]).split(",")[0]}' for label in unique_labels], mean_volumes, 
                   yerr=std_volumes, capsize=3, alpha=0.7)
    plt.title(f'Mean Volume by Label (Step {step})')
    plt.ylabel('Mean Gramian Volume')
    plt.xticks(rotation=90)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean_vol) in enumerate(zip(bars, mean_volumes)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_volumes[i] + 0.001,
                f'{mean_vol:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/volume_statistics_step_{step}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print some statistics
    print("\nVolume Statistics by Label:")
    print("=" * 50)
    for label in unique_labels:
        vols = volumes_by_label[label]
        print(f"{(id_to_caption[label]).split(',')[0]}: n={len(vols):2d}, mean={np.mean(vols):.4f}, std={np.std(vols):.4f}, "
              f"min={np.min(vols):.4f}, max={np.max(vols):.4f}")
    
    return volumes_by_label




def create_embedding_progression_gif(save_dir, output_path="embedding_progression.gif"):
    """Create animated GIF showing embedding evolution during training."""
    import glob
    from PIL import Image
    
    # Find all tsne plots
    pattern = f"{save_dir}/tsne_modalities_step_*.png"
    image_files = sorted(glob.glob(pattern), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    if len(image_files) < 2:
        print("Not enough images to create progression GIF")
        return
    
    images = []
    for file in image_files:
        img = Image.open(file)
        images.append(img)
    
    # Save as GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=800,  # milliseconds per frame
        loop=0
    )
    
    print(f"Embedding progression GIF saved to {output_path}")