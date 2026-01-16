import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
import torch.distributed as dist

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment as linear_assignment

import colorcet as cc

from easydict import EasyDict as edict

from utils.distributed import all_gather_list, ddp_allgather
from utils.volume import volume_computation3
from dataset_EEG.name_map_ID import id_to_caption


DEFAULT_TSNE_KWARGS = dict(
    n_components=2,
    perplexity=30,
    learning_rate='auto',
    init='pca',
    random_state=42,
    max_iter=1000,
)
MODALITY_MARKERS = {'Text': '*', 'Image': 's', 'EEG': '^'}
PAIRWISE_CONFIGS = [
    ("Text-Image", "text_image", 'text', 'vision', ('o', 's')),
    ("Text-EEG", "text_eeg", 'text', 'eeg', ('o', '^')),
    ("Image-EEG", "image_eeg", 'vision', 'eeg', ('s', '^')),
]


def create_palette(size):
    base = list(cc.palette['glasbey'])
    #how many times to repeat the base palette to reach size
    repeats = int(np.ceil(size / len(base)))
    #return the base list of colors, repeated and truncated at length=size
    return (base * repeats)[:size]


def run_tsne(embeddings, **overrides):
    #convenience function to run tsne with default params plus any overrides
    params = {**DEFAULT_TSNE_KWARGS, **overrides}
    return TSNE(**params).fit_transform(embeddings)


def get_label_name(label):
    return id_to_caption.get(label, str(label)).split(',')[0]


def creat_label_colors(labels):
    unique_labels = sorted(np.unique(labels))
    #build a palette of distinct colors using glasbey palette
    palette = create_palette(len(unique_labels))
    #return a dict mapping each label to a color
    return {label: palette[idx] for idx, label in enumerate(unique_labels)}


def scatter_plot_modalities(
    ax: plt.Axes,
    embeddings: np.ndarray,
    modality_labels: np.ndarray,
    class_labels: np.ndarray,
    label_colors: dict[int, str],
    markers: dict[str, str],
    size: int = 50,
):
    unique_labels = sorted(label_colors)
    for modality, marker in markers.items():
        modality_mask = modality_labels == modality
        for label in unique_labels:
            mask = modality_mask & (class_labels == label)
            if not np.any(mask):
                continue
            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                marker=marker,
                c=[label_colors[label]],
                alpha=0.7,
                s=size,
            )


def add_legends(ax: plt.Axes, markers: dict[str, str], label_colors: dict[int, str]):
    ax.axis('off')
    modality_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color='black',
            linestyle='None',
            markersize=10,
            label=modality,
        )
        for modality, marker in markers.items()
    ]
    label_handles = [
        Rectangle((0, 0), 1, 1, facecolor=color, label=f'{label}-{get_label_name(label)}')
        for label, color in label_colors.items()
    ]
    legend1 = ax.legend(handles=modality_handles, title="Modalities", loc='upper left', fontsize=11)
    ax.add_artist(legend1)
    ax.legend(
        handles=label_handles,
        title="Labels",
        loc='center left',
        fontsize=11,
        ncol=2 if len(label_handles) > 20 else 1,
    )


def pick_connection_indices(labels: np.ndarray) -> np.ndarray:
    indices = []
    for label in np.unique(labels):
        class_indices = np.where(labels == label)[0]
        if class_indices.size:
            indices.append(class_indices[0])
    return np.array(indices, dtype=int)


def draw_alignment_lines(ax: plt.Axes, embeddings: np.ndarray, indices: np.ndarray, n_samples: int):
    for idx in indices:
        ax.plot(
            [embeddings[idx, 0], embeddings[idx + n_samples, 0]],
            [embeddings[idx, 1], embeddings[idx + n_samples, 1]],
            'k-',
            alpha=0.4,
            linewidth=0.7,
        )
        ax.plot(
            [embeddings[idx, 0], embeddings[idx + 2 * n_samples, 0]],
            [embeddings[idx, 1], embeddings[idx + 2 * n_samples, 1]],
            'k-',
            alpha=0.4,
            linewidth=0.7,
        )
        ax.plot(
            [embeddings[idx + n_samples, 0], embeddings[idx + 2 * n_samples, 0]],
            [embeddings[idx + n_samples, 1], embeddings[idx + 2 * n_samples, 1]],
            'k-',
            alpha=0.4,
            linewidth=0.7,
        )


def compute_centroids(feats: Tensor, labels: torch.Tensor, num_classes: int) -> Tensor:
    centroids = []
    for idx in range(num_classes):
        mask = labels == idx
        if not torch.any(mask):
            centroids.append(torch.zeros((1, feats.shape[1]), device=feats.device))
            continue
        centroids.append(feats[mask].mean(dim=0, keepdim=True))
    return torch.cat(centroids, dim=0)


def cosine_similarity_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom = np.where(denom == 0, 1e-12, denom)
    return np.sum(a * b, axis=1) / denom


class TsnePlot:
    def __init__(self, perplexity=30, learning_rate='auto', n_iter=1000, random_state=42):
        self.tsne_params = {
            **DEFAULT_TSNE_KWARGS,
            'perplexity': perplexity,
            'learning_rate': learning_rate,
            'max_iter': n_iter,
            'random_state': random_state,
        }

    def plot(self, embedding: np.ndarray, labels: np.ndarray, score: float, output_dir: str, step: str):
        reduced_embedding = run_tsne(embedding, **self.tsne_params)

        max_val = np.max(reduced_embedding)
        min_val = np.min(reduced_embedding)
        reduced_embedding = (reduced_embedding - min_val) / (max_val - min_val + 1e-12)

        label_colors = creat_label_colors(labels)

        plt.figure(figsize=(12, 10))
        ax = plt.gca()
        ax.tick_params(axis='both', labelsize=11)

        for label in np.unique(labels):
            mask = labels == label
            ax.scatter(
                reduced_embedding[mask, 0],
                reduced_embedding[mask, 1],
                c=[label_colors[label]],
                label=f'{label}-{get_label_name(label)}',
                alpha=0.6,
            )

        ax.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0, ncol=2)
        plt.savefig(
            f'{output_dir}/tsne_eeg_kmean_{score:.5f}_step_{step}.png',
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()
        return reduced_embedding


class K_means:
    def __init__(self, n_clusters=40, random_state=45):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def transform(self, embed: np.ndarray, gt_labels: np.ndarray) -> float:
        pred_labels = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit_predict(embed)
        return self.cluster_acc(gt_labels, pred_labels)

    def cluster_acc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum(w[i, j] for i, j in zip(*ind)) * 1.0 / y_pred.size


@torch.no_grad()
def create_plots(model, val_loader, args, tasks):
    feat_t, feat_v, feat_e = [], [], []
    label_tensors, subject_ids = [], []

    main_task = tasks.split('%')[0]

    for _, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
        batch = edict(batch)
        outputs = model(batch, main_task, compute_loss=False)

        feat_t.append(outputs['feat_t'])
        feat_v.append(outputs['feat_v'])
        feat_e.append(outputs['feat_e'])

        if 'labels' in batch:
            label_tensors.append(batch['labels'])

        if 'eeg_subjects' in batch:
            subject_ids.extend(batch['eeg_subjects'].cpu().numpy().tolist())

    feat_t = ddp_allgather(torch.cat(feat_t, dim=0))
    feat_v = ddp_allgather(torch.cat(feat_v, dim=0))
    feat_e = ddp_allgather(torch.cat(feat_e, dim=0))

    labels_array = np.array([])
    if label_tensors:
        labels_tensor = ddp_allgather(torch.cat(label_tensors, dim=0))
        labels_array = labels_tensor.cpu().numpy()

    subject_ids = [id for batch in all_gather_list(subject_ids) for id in batch]

    if dist.get_rank() != 0:
        return

    #used for naming the files nd plots
    step = args.run_cfg.checkpoint.split('_')[-1].replace('.pt', '')

    tsne_data = {
        'text': feat_t.cpu().numpy(),
        'vision': feat_v.cpu().numpy(),
        'eeg': feat_e.cpu().numpy(),
        'subjects': np.array(subject_ids[:feat_t.shape[0]]) if subject_ids else np.array([]),
        'labels': labels_array[:feat_t.shape[0]],
    }

    plot_tsne_eeg_embeddings({'eeg': tsne_data['eeg'], 'labels': tsne_data['labels']}, args.run_cfg.output_dir, step)
    plot_tsne_all_modalities(tsne_data, args.run_cfg.output_dir, step)
    plot_tsne_pairwise_modalities(tsne_data, args.run_cfg.output_dir, step)
    plot_alignment_heatmap(tsne_data, args.run_cfg.output_dir, step)
    plot_volume_heatmap_per_sample(tsne_data, args.run_cfg.output_dir, step)
    plot_tsne_centroids(
        feat_t,
        feat_v,
        feat_e,
        labels_array,
        output_dir=args.run_cfg.output_dir,
        title=f'Centroids - step {step}',
        step=step,
    )


@torch.no_grad()
def plot_tsne_eeg_embeddings(data, output_dir, step):
    k_means = K_means(n_clusters=40)
    clustering_acc = k_means.transform(data["eeg"], data["labels"])
    print(f"[Test KMeans score Proj: {clustering_acc}]")
    TsnePlot().plot(data["eeg"], data["labels"], clustering_acc, output_dir, step)

#plot all modalities together
@torch.no_grad()
def plot_tsne_all_modalities(data, output_dir, step):
    labels = np.asarray(data['labels'])
    if labels.size == 0:
        return

    label_colors = creat_label_colors(labels)
    n_samples = data['text'].shape[0]

    stacked = np.vstack([data['text'], data['vision'], data['eeg']])
    modality_labels = np.array(['Text'] * n_samples + ['Image'] * n_samples + ['EEG'] * n_samples)
    class_labels = np.tile(labels, 3) #repeat labels for each modality

    embeddings_2d = run_tsne(stacked)

    def build_plot(path: str, title: str, alignment_lines: bool = False):
        fig, (ax_plot, ax_leg) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [3, 1]})
        scatter_plot_modalities(ax_plot, embeddings_2d, modality_labels, class_labels, label_colors, MODALITY_MARKERS)
        ax_plot.set_title(title)
        ax_plot.set_xlabel('t-SNE Dimension 1')
        ax_plot.set_ylabel('t-SNE Dimension 2')
        ax_plot.grid(True, alpha=0.3)
        
        if alignment_lines:
            connection_indices = pick_connection_indices(labels)
            draw_alignment_lines(ax_plot, embeddings_2d, connection_indices, n_samples)

        add_legends(ax_leg, MODALITY_MARKERS, label_colors)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    build_plot(f'{output_dir}/tsne_all_modalities_step_{step}.png', f'Multimodal Embedding Space (Step {step})')
    #same plot but with alignment lines
    build_plot(
        f'{output_dir}/tsne_all_modalities_alignment_step_{step}.png',
        f'Multimodal Alignment Embedding Space (Step {step})',
        alignment_lines=True,
    )
    print(f"t-SNE visualizations saved to {output_dir}/{step}")

#plot modality pairs
@torch.no_grad()
def plot_tsne_pairwise_modalities(data, output_dir, step=0):
    labels = np.asarray(data['labels'])

    label_colors = creat_label_colors(labels)

    for title, suffix, first_key, second_key, markers in PAIRWISE_CONFIGS:
        feat1 = data[first_key]
        feat2 = data[second_key]
        combined = np.vstack([feat1, feat2])
        embedded = run_tsne(combined)
        n_samples = feat1.shape[0]

        fig, (ax_plot, ax_leg) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [3, 1]})
        for label in sorted(label_colors):
            indices = np.where(labels == label)[0]
            if not indices.size:
                continue
            ax_plot.scatter(
                embedded[indices, 0],
                embedded[indices, 1],
                c=[label_colors[label]],
                marker=markers[0],
                alpha=0.7,
                s=50,
            )
            ax_plot.scatter(
                embedded[n_samples + indices, 0],
                embedded[n_samples + indices, 1],
                c=[label_colors[label]],
                marker=markers[1],
                alpha=0.7,
                s=50,
            )
        ax_plot.set_title(f'{title} Alignment')
        ax_plot.grid(True, alpha=0.3)

        add_legends(ax_leg, {title.split('-')[0]: markers[0], title.split('-')[1]: markers[1]}, label_colors)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/tsne_pairwise_{suffix}_step_{step}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)


@torch.no_grad()
def plot_tsne_centroids(
    text_feats: Tensor,
    image_feats: Tensor,
    eeg_feats: Tensor,
    labels: np.ndarray,
    output_dir: str = 'centroids.png',
    random_state: int = 42,
    title: str = 'Multimodal Latent Space',
    annotate_centroids: bool = True,
    step: str | int = 0,
):

    labels_tensor = torch.as_tensor(labels).long().cpu()
    num_classes = int(labels_tensor.max().item()) + 1
    label_colors = creat_label_colors(np.arange(num_classes))

    centroid_t = compute_centroids(text_feats.cpu(), labels_tensor, num_classes)
    centroid_v = compute_centroids(image_feats.cpu(), labels_tensor, num_classes)
    centroid_e = compute_centroids(eeg_feats.cpu(), labels_tensor, num_classes)

    all_centroids = torch.cat([centroid_t, centroid_v, centroid_e], dim=0).numpy()
    modality_labels = np.array(['Text'] * num_classes + ['Image'] * num_classes + ['EEG'] * num_classes)
    class_labels = np.tile(np.arange(num_classes), 3)
    emb2d = run_tsne(all_centroids, random_state=random_state)

    fig, (ax_plot, ax_leg) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [3, 1]})
    scatter_plot_modalities(
        ax_plot,
        emb2d,
        modality_labels,
        class_labels,
        label_colors,
        MODALITY_MARKERS,
        size=260,
    )
    if annotate_centroids:
        text_mask = modality_labels == 'Text'
        for idx, point in zip(np.arange(num_classes), emb2d[text_mask]):
            ax_plot.text(point[0], point[1], str(idx), ha='center', va='center', fontsize=11, color='k')

    ax_plot.set_title(f'{title} (Centroids Only)')
    ax_plot.set_xlabel('Dim 1')
    ax_plot.set_ylabel('Dim 2')
    ax_plot.grid(True, alpha=0.3)
    add_legends(ax_leg, MODALITY_MARKERS, label_colors)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/tsne_all_modalities_centroids_step_{step}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    for title, suffix, first_key, second_key, markers in PAIRWISE_CONFIGS:
        feats_map = {'text': centroid_t.numpy(), 'vision': centroid_v.numpy(), 'eeg': centroid_e.numpy()}
        combined = np.vstack([feats_map[first_key], feats_map[second_key]])
        embedded = run_tsne(combined, random_state=random_state)
        n_samples = feats_map[first_key].shape[0]

        fig, (ax_plot, ax_leg) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [3, 1]})
        for idx in range(num_classes):
            ax_plot.scatter(
                embedded[idx, 0],
                embedded[idx, 1],
                marker=markers[0],
                c=[label_colors[idx]],
                edgecolors='k',
                linewidths=0.6,
                s=260,
            )
            ax_plot.scatter(
                embedded[n_samples + idx, 0],
                embedded[n_samples + idx, 1],
                marker=markers[1],
                c=[label_colors[idx]],
                edgecolors='k',
                linewidths=0.6,
                s=260,
            )
            ax_plot.text(
                embedded[idx, 0],
                embedded[idx, 1],
                str(idx),
                ha='center',
                va='center',
                fontsize=11,
                color='k',
            )

        ax_plot.set_title(f'{title} Centroids')
        ax_plot.grid(True, alpha=0.3)
        add_legends(ax_leg, {title.split('-')[0]: markers[0], title.split('-')[1]: markers[1]}, label_colors)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/tsne_centroids_pairwise_{suffix}_step_{step}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    return output_dir

# kinda useless
@torch.no_grad()
def plot_alignment_heatmap(data, output_dir, step=0):
    labels = np.asarray(data['labels'])

    unique_labels = np.unique(labels)
    alignment_matrix = np.zeros((len(unique_labels), 4))

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        if mask.sum() < 1:
            continue

        t_class = data['text'][mask]
        v_class = data['vision'][mask]
        e_class = data['eeg'][mask]

        alignment_matrix[idx, 0] = cosine_similarity_batch(t_class, v_class).mean()
        alignment_matrix[idx, 1] = cosine_similarity_batch(t_class, e_class).mean()
        alignment_matrix[idx, 2] = cosine_similarity_batch(v_class, e_class).mean()

        t_tensor = torch.tensor(t_class, dtype=torch.float16)
        v_tensor = torch.tensor(v_class, dtype=torch.float16)
        e_tensor = torch.tensor(e_class, dtype=torch.float16)
        volumes = []
        for sample_idx in range(t_tensor.shape[0]):
            volume = volume_computation3(
                t_tensor[sample_idx : sample_idx + 1],
                v_tensor[sample_idx : sample_idx + 1],
                e_tensor[sample_idx : sample_idx + 1],
            )
            volumes.append(volume.item() if isinstance(volume, torch.Tensor) else volume)
        alignment_matrix[idx, 3] = np.mean(volumes)

    plt.figure(figsize=(10, max(6, len(unique_labels) * 0.3)))
    sns.heatmap(
        alignment_matrix,
        xticklabels=['Text-Image', 'Text-EEG', 'Image-EEG', 'Volume'],
        yticklabels=[get_label_name(label) for label in unique_labels],
        annot=True,
        fmt='.3f',
        cmap='viridis',
        center=0,
    )
    plt.title(f'Cross-Modal Alignment by Class (Step {step})')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/alignment_heatmap_step_{step}.png', dpi=300, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def plot_volume_heatmap_per_sample(data, output_dir, step=0):
    labels = np.asarray(data['labels'])

    feat_t = torch.tensor(data['text'], dtype=torch.float32)
    feat_v = torch.tensor(data['vision'], dtype=torch.float32)
    feat_e = torch.tensor(data['eeg'], dtype=torch.float32)

    volumes = []
    for idx in range(labels.shape[0]):
        volume = volume_computation3(
            feat_t[idx : idx + 1],
            feat_v[idx : idx + 1],
            feat_e[idx : idx + 1],
        )
        volumes.append(volume.item() if isinstance(volume, torch.Tensor) else volume)
    volumes = np.array(volumes)

    unique_labels = np.unique(labels)
    volumes_by_label = {label: volumes[labels == label] for label in unique_labels}
    max_samples = max(len(v) for v in volumes_by_label.values())

    volume_matrix = np.full((len(unique_labels), max_samples), np.nan)
    for row, label in enumerate(unique_labels):
        label_volumes = volumes_by_label[label]
        volume_matrix[row, : len(label_volumes)] = label_volumes

    plt.figure(figsize=(max(12, max_samples * 0.5), max(8, len(unique_labels) * 0.4)))
    sns.heatmap(
        volume_matrix,
        mask=np.isnan(volume_matrix),
        yticklabels=[f'{get_label_name(label)} (n={len(volumes_by_label[label])})' for label in unique_labels],
        xticklabels=[f'Sample {i+1}' for i in range(max_samples)],
        annot=True,
        fmt='.3f',
        cmap='viridis',
        cbar_kws={'label': 'Gramian Volume'},
    )
    plt.title(f'Volume Spanned by Embedding Triads per Sample (Step {step})')
    plt.xlabel('Sample Index within Label')
    plt.ylabel('Label (Class)')
    if max_samples > 20:
        tick_indices = np.linspace(0, max_samples - 1, min(20, max_samples), dtype=int)
        plt.xticks(tick_indices, [f'Sample {i+1}' for i in tick_indices], rotation=90)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/volume_heatmap_per_sample_step_{step}.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 8))
    import pandas as pd

    volume_data = []
    volume_labels = []
    for label in unique_labels:
        label_volumes = volumes_by_label[label]
        volume_data.extend(label_volumes)
        volume_labels.extend([get_label_name(label)] * len(label_volumes))
    df = pd.DataFrame({'Volume': volume_data, 'Label': volume_labels})

    plt.subplot(2, 1, 1)
    sns.boxplot(data=df, x='Label', y='Volume')
    plt.title(f'Distribution of Volumes by Label (Step {step})')
    plt.xticks(rotation=90)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    mean_volumes = [np.mean(volumes_by_label[label]) for label in unique_labels]
    std_volumes = [np.std(volumes_by_label[label]) for label in unique_labels]
    bars = plt.bar([get_label_name(label) for label in unique_labels], mean_volumes, yerr=std_volumes, capsize=3, alpha=0.7)
    plt.title(f'Mean Volume by Label (Step {step})')
    plt.ylabel('Mean Gramian Volume')
    plt.xticks(rotation=90)
    plt.grid(True, alpha=0.3)
    for bar, mean_vol, std_vol in zip(bars, mean_volumes, std_volumes):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std_vol + 0.001,
            f'{mean_vol:.3f}',
            ha='center',
            va='bottom',
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(f'{output_dir}/volume_statistics_step_{step}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nVolume Statistics by Label:")
    print("=" * 50)
    for label in unique_labels:
        vols = volumes_by_label[label]
        print(
            f"{get_label_name(label)}: n={len(vols):2d}, mean={np.mean(vols):.4f}, std={np.std(vols):.4f}, "
            f"min={np.min(vols):.4f}, max={np.max(vols):.4f}"
        )

# unused
@torch.no_grad()
def create_embedding_progression_gif(save_dir, output_path="embedding_progression.gif"):
    """Create animated GIF showing embedding evolution during training."""
    import glob
    from PIL import Image

    pattern = f"{save_dir}/tsne_modalities_step_*.png"
    image_files = sorted(glob.glob(pattern), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if len(image_files) < 2:
        print("Not enough images to create progression GIF")
        return

    images = [Image.open(file) for file in image_files]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=800,
        loop=0,
    )