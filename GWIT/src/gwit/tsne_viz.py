import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch
from tqdm import tqdm
import os

def plot_tsne_eeg(model, eval_dataloader, args, step=0):
    """
    Create t-SNE visualization of multimodal embeddings from GRAM model.
    
    Args:
        model: Your GRAM model
        eval_dataloader: DataLoader for evaluation data
        args: Training arguments
        save_dir: Directory to save plots
        step: Current training step (for filename)
    """
    model.eval()

    # Define labels to visualize
    labels_to_keep = [4, 3, 15]  # 3 most numerous labels
    label_names = {4: 'espresso maker', 3: 'anemone fish', 15: 'cellular telephone'}
    label_colors = {4: '#ff6d60', 3: '#19b99c', 15: '#bf00ff'} 
    embedding_marker = {'Text': '*', 'Image': 's', 'EEG': '^'} 
    # Collect embeddings and metadata
    feat_t_list = []
    feat_i_list = []
    feat_e_list = []
    subject_ids = []
    labels_list = []
    
    print("Collecting embeddings...")
    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            # Filter batch to keep only my labels
            labels = batch['labels']
            mask = torch.tensor([label in labels_to_keep for label in labels])
            
            if not mask.any():
                continue # Skip batch if none of the labels are in the batch

            # apply the mask to the batch
            filtered_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and value.shape[0] == len(labels):
                    filtered_batch[key] = value[mask]
                elif isinstance(value, list) and len(value) == len(labels):
                    filtered_batch[key] = [value[i] for i in range(len(value)) if mask[i]]
                else:
                    filtered_batch[key] = value

            # Get embeddings of the filtered batch
            feat_t = model.batch_get(filtered_batch, 'feat_t').cpu().numpy()
            feat_i = model.batch_get(filtered_batch, 'feat_i').cpu().numpy()
            feat_e = model.batch_get(filtered_batch, 'feat_e').cpu().numpy()
            
            feat_t_list.append(feat_t)
            feat_i_list.append(feat_i)
            feat_e_list.append(feat_e)
            
            # Collect metadata
            labels_list.extend(filtered_batch['labels'])
            if args.subject_num == 0: # Using all subjects
                subject_ids.extend(filtered_batch['eeg_subjects'].cpu().numpy())
            

    # Concatenate all embeddings
    feat_t = np.concatenate(feat_t_list, axis=0)
    feat_i = np.concatenate(feat_i_list, axis=0)
    feat_e = np.concatenate(feat_e_list, axis=0)
    
    print(f"Collected embeddings: Text={feat_t.shape}, Image={feat_i.shape}, EEG={feat_e.shape}")
    
    # Create combined embedding matrix
    all_embeddings = np.concatenate([feat_t, feat_i, feat_e], axis=0)
    
    # Create modality labels
    n_samples = feat_t.shape[0]
    modality_labels = ['Text'] * n_samples + ['Image'] * n_samples + ['EEG'] * n_samples
    class_labels = labels_list * 3  # Repeat labels for each modality
    # Create sample indices (for matching across modalities)
    sample_indices = list(range(n_samples)) * 3
    
    print(f"Running t-SNE on {all_embeddings.shape[0]} embeddings...")
    
    # Run t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, all_embeddings.shape[0] // 4),
        random_state=42,
        n_iter=1000,
        learning_rate='auto',
        init='pca'
    )
    
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # 1. Basic modality visualization
    plt.figure(figsize=(12, 8))
    colors = {'Text': "#ff6d60", 'Image': '#19b99c', 'EEG': '#bf00ff'}

    # Plot all the combinations of modalities and classes
    for modality in ['Text', 'Image', 'EEG']:
        for label in labels_to_keep:
            modality_mask = np.array(modality_labels) == modality
            class_mask = np.array(class_labels) == label
            mask = modality_mask & class_mask
            
            label_name = label_names.get(label, f'Label {label}')
            plt.scatter(
                embeddings_2d[mask, 0], 
                embeddings_2d[mask, 1],
                marker = embedding_marker[modality],
                c = label_colors[label], 
                label=f'{modality} - {label_name}', 
                alpha=0.7,
                s=50
            )
    
    plt.title(f'Multimodal Embedding Space (Step {step})')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/tsne_modalities_step_{step}.png', dpi=300, bbox_inches='tight')
    plt.close()
    

    ## 2. Alignment visualization (connecting matching samples)
    #plt.figure(figsize=(12, 8))
    #
    ## Plot points
    #for modality in ['Text', 'Image', 'EEG']:
    #    mask = np.array(modality_labels) == modality
    #    plt.scatter(
    #        embeddings_2d[mask, 0], 
    #        embeddings_2d[mask, 1],
    #        c=colors[modality], 
    #        label=modality, 
    #        alpha=1.0,
    #        s=60
    #    )
    #
    ## Draw lines connecting matching samples (sample every nth to avoid clutter)
    #n_connections = min(50, n_samples)  # Limit connections to avoid clutter
    #connection_indices = np.linspace(0, n_samples-1, n_connections, dtype=int)
    #
    #for i in connection_indices:
    #    text_point = embeddings_2d[i]
    #    image_point = embeddings_2d[i + n_samples]
    #    eeg_point = embeddings_2d[i + 2*n_samples]
    #    
    #    # Draw triangle connecting the three modalities for this sample
    #    plt.plot([text_point[0], image_point[0]], [text_point[1], image_point[1]], 
    #            'k-', alpha=0.2, linewidth=0.5)
    #    plt.plot([text_point[0], eeg_point[0]], [text_point[1], eeg_point[1]], 
    #            'k-', alpha=0.2, linewidth=0.5)
    #    plt.plot([image_point[0], eeg_point[0]], [image_point[1], eeg_point[1]], 
    #            'k-', alpha=0.2, linewidth=0.5)
    #
    #plt.title(f'Multimodal Alignment Visualization (Step {step})')
    #plt.xlabel('t-SNE Dimension 1')
    #plt.ylabel('t-SNE Dimension 2')
    #plt.legend()
    #plt.grid(True, alpha=0.3)
    #plt.tight_layout()
    #plt.savefig(f'{args.output_dir}/tsne_alignment_step_{step}.png', dpi=300, bbox_inches='tight')
    #plt.close()
    #
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
    #    plt.savefig(f'{args.output_dir}/tsne_subjects_step_{step}.png', dpi=300, bbox_inches='tight')
    #    plt.close()
    #
    # 4. Compute and visualize alignment metrics
    alignment_metrics = compute_alignment_metrics(feat_t, feat_i, feat_e)
    
    plt.figure(figsize=(12, 8))
    metrics_names = list(alignment_metrics.keys())
    metrics_values = list(alignment_metrics.values())
    
    bars = plt.bar(metrics_names, metrics_values, color=['#ff6d60', '#19b99c', '#bf00ff'])
    plt.title(f'Modality Alignment Metrics (Step {step})')
    plt.ylabel('Cosine Similarity')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/alignment_metrics_step_{step}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"t-SNE visualizations saved to {args.output_dir}")
    print(f"Alignment metrics: {alignment_metrics}")
    
    #return alignment_metrics, embeddings_2d

def compute_alignment_metrics(feat_t, feat_i, feat_e):
    """Compute alignment metrics between modalities."""
    # Compute mean cosine similarities
    def mean_cosine_similarity(x, y):
        # Normalize
        x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)
        y_norm = y / np.linalg.norm(y, axis=1, keepdims=True)
        # Compute diagonal of similarity matrix (matching pairs)
        return np.mean(np.sum(x_norm * y_norm, axis=1))
    
    metrics = {
        'Text-Image': mean_cosine_similarity(feat_t, feat_i),
        'Text-EEG': mean_cosine_similarity(feat_t, feat_e),
        'Image-EEG': mean_cosine_similarity(feat_i, feat_e)
    }
    
    return metrics

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

# Example usage in your training loop:
def integrate_with_training():
    """
    Example of how to integrate this into your training loop.
    Add this to your pretrain_eeg_gram.py file.
    """
    code_example = '''
    # In your training loop, replace the existing t-SNE call with:
    
    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
        evaluate_pretraining(gram_model, args, global_step, accelerator, eval_dataset, eval_dataloader)
        
        # Add this line for t-SNE visualization
        alignment_metrics, tsne_coords = plot_tsne_eeg(
            gram_model, eval_dataloader, args, 
            save_dir=f"{args.output_dir}/tsne_plots", 
            step=global_step
        )
        
        # Log alignment metrics to wandb
        if accelerator.is_main_process:
            wandb.log({
                "alignment/text_image": alignment_metrics['Text-Image'],
                "alignment/text_eeg": alignment_metrics['Text-EEG'], 
                "alignment/image_eeg": alignment_metrics['Image-EEG']
            }, step=global_step)
        
        gram_model.train()
    
    # At the end of training, create progression GIF
    if accelerator.is_main_process:
        create_embedding_progression_gif(f"{args.output_dir}/tsne_plots")
    '''
    return code_example