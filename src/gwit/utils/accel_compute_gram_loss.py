import os
os.environ['HF_HOME'] = './cache/'
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import LayerNorm as LayerNorm
import torch.utils.checkpoint
from utils.distributed import all_gather_with_grad, concat_all_gather

from gram_utils import volume_computation3

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

class Match_head(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.activation = GELU()
        self.layernorm = LayerNorm(hidden_size, eps=1e-12)
        self.linear2 = nn.Linear(hidden_size, 2)
    def forward(self, cls_token):
        return self.linear2(self.layernorm(self.activation(self.linear1(cls_token))))
    
class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, text_features, condition_features):
        # text_features: [batch_size, seq_length, hidden_size]
        # condition_features: [batch_size, condition_seq_length, hidden_size]
        attended_features, _ = self.attention(
            text_features.transpose(0, 1),
            condition_features.transpose(0, 1),
            condition_features.transpose(0, 1)
        )
        attended_features = attended_features.transpose(0, 1)
        return self.layer_norm(text_features + attended_features)

# Trying to replicate the get_multimodal_forward_input_vision behaviour but for images
def get_multimodal_forward_input_image(latents, hidden_trans_image_multimodal, accelerator):
    # Shape from vae is [b, 4, 64, 64]
    b, c, h, w = latents.shape
    # Reshape to [b, (h*w), c] to treat spatial positions as a sequence
    # This converts from [b, 4, 64, 64] to [b, 4096, 4]
    image_output = latents.permute(0, 2, 3, 1).reshape(b, h*w, c)
    
    # Transform features to multimodal dimension
    image_output = hidden_trans_image_multimodal(image_output.to(accelerator.device) )
    
    return image_output  # Shape: [b, 4096, multimodal_dim]

# Really is the same as the image one, since the EEG encoder gives the same shape as the VAE
def get_multimodal_forward_input_eeg(eeg_embedding, hidden_trans_eeg_multimodal):
    # EEG encoder gives shape [b, 320, 64, 64]
    print("eeg_embedding shape: ", eeg_embedding.shape)
    b, c, h, w = eeg_embedding.shape
    
    # Permute and reshape to treat spatial positions as a sequence
    eeg_embedding = eeg_embedding.permute(0, 2, 3, 1).reshape(b, h*w, c) 
    
    # Transform features to multimodal dimension
    eeg_embedding = hidden_trans_eeg_multimodal(eeg_embedding)
    
    return eeg_embedding  # Shape: [b, 4096, multimodal_dim]


def accel_compute_gram_loss(latents, encoder_hidden_states, eeg_embedding, batch,
                      latents_proj, text_proj, controlnet_image_proj, temperature, accelerator, 
                      hidden_trans_image_multimodal, hidden_trans_eeg_multimodal, text_encoder):
    itm_head = Match_head(1024).to(accelerator.device)
    ### --------- ###
    # POOLING
    pooled_latents = latents.mean(dim=[2,3]) #should be [b,4]
    # With CLIP the eos token has a similar role to the CLS token in BERT, so we take it as the pooled representation
    pooled_captions = encoder_hidden_states[:,-1,:] #should be [b, 1024]
    pooled_eeg_embedding = eeg_embedding[:,:,0] 
    # IS this mean necessary? TODO
    pooled_eeg_embedding = torch.mean(pooled_eeg_embedding, dim=1) #should be [b,128]

    #Projections
    projected_latents = latents_proj(pooled_latents) #should be [b,128]
    projected_captions = text_proj(pooled_captions) #should be [b,128]
    projected_controlnet_image = controlnet_image_proj(pooled_eeg_embedding) #should be [b,128]

    # normalize
    projected_latents = F.normalize(projected_latents, dim=-1) #feat_v
    projected_captions = F.normalize(projected_captions, dim=-1) #feat_t
    projected_controlnet_image = F.normalize(projected_controlnet_image, dim=-1) #feat_a

    gathered_latents = concat_all_gather(projected_latents) #feat_v_all
    gathered_captions = concat_all_gather(projected_captions) #feat_t_all
    gathered_eeg = concat_all_gather(projected_controlnet_image) #feat_a_all
                
    input_ids = batch['input_ids']
    print("input_ids shape: ", input_ids.shape)
    input_ids_collate = concat_all_gather(input_ids)
    print("input_ids_collate shape: ", input_ids_collate.shape)

    attention_mask = batch['attention_mask']
    print("attention_mask shape: ", attention_mask.shape)
    attention_mask_collate = concat_all_gather(attention_mask)
    print("attention_mask_collate shape: ", attention_mask_collate.shape)

    # volume computation
    gram_volume = volume_computation3(projected_captions, gathered_latents, gathered_eeg) #volume_computation3(projected_captions, combined_latents, combined_eeg)
    gram_volume = gram_volume / temperature
    gram_volumeT = volume_computation3(gathered_captions, projected_latents, projected_controlnet_image).T #volume_computation3(combined_captions, projected_latents, projected_controlnet_image).T
    gram_volumeT = gram_volumeT / temperature
                
    # Compute gram losses:
    batch_size = projected_captions.size(0)
    rank = accelerator.process_index
    print("batch_size: ", batch_size)
    print("gram_volume shape: ", gram_volume.shape)
    print("gram_volumeT shape: ", gram_volumeT.shape)
    print("Diagonal values of gram_volume:", torch.diagonal(gram_volume[:batch_size, :batch_size]))
    print("Mean diagonal value:", torch.diagonal(gram_volume[:batch_size, :batch_size]).mean())
    print("Mean off-diagonal value:", (gram_volume[:batch_size, :batch_size].sum() - torch.diagonal(gram_volume[:batch_size, :batch_size]).sum()) / (batch_size * (batch_size - 1)))
    targets = torch.linspace(rank * batch_size, rank * batch_size + batch_size - 1, batch_size, dtype=int).to(accelerator.device)

    loss_d2a = F.cross_entropy(-gram_volume, targets, label_smoothing=0.1)                
    loss_a2d = F.cross_entropy(-gram_volumeT, targets, label_smoothing=0.1) 

    gram_loss = (loss_d2a + loss_a2d) / 2
    print("\n projected images and text cosine similarity",F.cosine_similarity(projected_latents, projected_captions, dim=-1).mean().detach().item())
    print("\n Gram Loss: ", gram_loss, gram_loss.shape)


    '''
    DAM Loss
    As far as I understand this line
    condition_feats = self.batch_get(batch, f'condition_feats_va')
        condition_feats_v = self.batch_get(batch, 'condition_feats_v')
            vision_output = self.batch_get(batch, 'vision_output')
                 vision_output = self.forward_vision_encoder(batch.vision_pixels)
    As far as I understand, the forward_vision_encoder just calls the forward method of the vision encoder. Since I have the latent images, I should already have the corresponding data.
            condition_feats_v = self.get_multimodal_forward_input_vision(vision_output)
    This line calls a function that I THINK prepares the vision_output to be part of the hidden states parameter of the multimodal encoder (i.e. the text encoder).
    It applies a hidden transformation and a reshape (check general_module.py)
    I need to understand what shapes are in play here. TODO
        condition_feats_a = self.batch_get(batch, 'condition_feats_a')
    The code then repeats the same steps for the audio data.
    Then condition_feats_v and condition_feats_a are concatenated (torch.cat).
    '''
    condition_feats_img = latents
    condition_feats_img = get_multimodal_forward_input_image(condition_feats_img, hidden_trans_image_multimodal, accelerator) #condition_feats_v
    print("condition_feats_img shape: ", condition_feats_img.shape)
    condition_feats_eeg = eeg_embedding
    condition_feats_eeg = get_multimodal_forward_input_eeg(condition_feats_eeg, hidden_trans_eeg_multimodal) #condition_feats_a
    print("condition_feats_eeg shape: ", condition_feats_eeg.shape)

    condition_feats_img_eeg = torch.cat((condition_feats_img, condition_feats_eeg),dim=1) #condition_feats
    print("condition_feats_img_eeg shape: ", condition_feats_img_eeg.shape)

    '''
    condition_feats_collate = all_gather_with_grad(condition_feats)
    '''
    condition_feats_collate = all_gather_with_grad(condition_feats_img_eeg)
    print("condition_feats_collate shape: ", condition_feats_collate.shape)
    

    # From original code
    with torch.no_grad():
        weights_t2cond = F.softmax(-(gram_volume), dim=1) + 1e-4
        print("weights_t2cond shape: ", weights_t2cond.shape)
        weights_t2cond[:, rank * batch_size : rank * batch_size + batch_size].fill_diagonal_(0)

        weights_cond2t = F.softmax(-(gram_volumeT), dim=1) + 1e-4
        weights_cond2t[:, rank * batch_size : rank * batch_size + batch_size].fill_diagonal_(0)

    condition_feats_neg = []
    for b in range(batch_size):
        neg_idx = torch.multinomial(weights_t2cond[b], 1).item()
        condition_feats_neg.append(condition_feats_collate[neg_idx])
    condition_feats_neg = torch.stack(condition_feats_neg, dim=0)

    text_ids_neg = []
    text_atts_neg = []
    for b in range(batch_size):
        neg_idx = torch.multinomial(weights_cond2t[b], 1).item()
        text_ids_neg.append(input_ids_collate[neg_idx])
        text_atts_neg.append(attention_mask_collate[neg_idx])

    text_ids_neg = torch.stack(text_ids_neg, dim=0)
    text_atts_neg = torch.stack(text_atts_neg, dim=0)

    input_ids_1 = torch.cat((input_ids, input_ids, text_ids_neg),dim=0)
    attention_mask_1 = torch.cat((attention_mask, attention_mask, text_atts_neg),dim=0)
    
    condition_feats = torch.cat((condition_feats_img_eeg,condition_feats_neg,condition_feats_img_eeg),dim=0)

    cross_attention = CrossAttentionLayer(hidden_size=1024).to(accelerator.device)
    output = text_encoder(input_ids=input_ids_1, attention_mask=attention_mask_1)
    cross_attended_output = cross_attention(output.last_hidden_state, condition_feats)


    batch_size = condition_feats_neg.shape[0]
    logits = itm_head(cross_attended_output[:,0]) #.half()) TODO why half?
    ground_truth = torch.zeros(batch_size*3).long().cuda()
    ground_truth[:batch_size] = 1
    loss_dam = F.cross_entropy(logits,ground_truth) #itm (dtm)
    print("loss_dam: ", loss_dam)
    #dam_loss = (0.1 * loss)
    
    print("Logging for dam loss:")
    # Classification metrics
    with torch.no_grad():
        # Get predictions
        predictions = torch.argmax(logits, dim=1)
        # Calculate accuracy
        accuracy = (predictions == ground_truth).float().mean()
        # Separate accuracy for positive and negative pairs
        pos_acc = (predictions[:batch_size] == ground_truth[:batch_size]).float().mean()
        neg_acc1 = (predictions[batch_size:2*batch_size] == ground_truth[batch_size:2*batch_size]).float().mean()
        neg_acc2 = (predictions[2*batch_size:] == ground_truth[2*batch_size:]).float().mean()

        print(f"DAM Classification Metrics:")
        print(f"  Total Accuracy: {accuracy.item():.4f}")
        print(f"  Positive Pair Accuracy: {pos_acc.item():.4f}")
        print(f"  Negative Pair Type 1 Accuracy: {neg_acc1.item():.4f}")
        print(f"  Negative Pair Type 2 Accuracy: {neg_acc2.item():.4f}")
        
        # Logit distribution
        pos_logits = logits[:batch_size, 1] - logits[:batch_size, 0]  # Higher value = more confident positive
        neg1_logits = logits[batch_size:2*batch_size, 1] - logits[batch_size:2*batch_size, 0]
        neg2_logits = logits[2*batch_size:, 1] - logits[2*batch_size:, 0]

        print(f"Logit Distribution:")
        print(f"  Positive pairs - Mean: {pos_logits.mean().item():.4f}, Std: {pos_logits.std().item():.4f}")
        print(f"  Negative pairs (condition neg) - Mean: {neg1_logits.mean().item():.4f}, Std: {neg1_logits.std().item():.4f}")
        print(f"  Negative pairs (text neg) - Mean: {neg2_logits.mean().item():.4f}, Std: {neg2_logits.std().item():.4f}")
        print(f"  Margin (pos-neg): {(pos_logits.mean() - torch.cat([neg1_logits, neg2_logits]).mean()).item():.4f}")

        # Negative mininq quality
        # Get gram matrix values for selected negatives
        neg_indices_cond = [torch.multinomial(weights_t2cond[b, :min(weights_t2cond.shape[1], condition_feats_collate.shape[0])], 1).item() for b in range(batch_size)]
        neg_indices_text = [torch.multinomial(weights_cond2t[b, :batch_size], 1).item() for b in range(batch_size)]
        
        # Get values from gram matrices for these indices
        neg_values_cond = torch.tensor([gram_volume[b, idx] for b, idx in enumerate(neg_indices_cond)])
        neg_values_text = torch.tensor([gram_volumeT[b, idx] for b, idx in enumerate(neg_indices_text)])
        
        # Compare with random sampling
        random_indices_cond = torch.randint(0, min(weights_t2cond.shape[1], condition_feats_collate.shape[0]), (batch_size,))
        random_indices_text = torch.randint(0, batch_size, (batch_size,))
        
        random_values_cond = torch.tensor([gram_volume[b, idx] for b, idx in enumerate(random_indices_cond)])
        random_values_text = torch.tensor([gram_volumeT[b, idx] for b, idx in enumerate(random_indices_text)])
        
        print(f"Negative Mining Quality:")
        print(f"  DAM neg condition mean value: {neg_values_cond.mean().item():.4f}")
        print(f"  Random neg condition mean value: {random_values_cond.mean().item():.4f}")
        print(f"  DAM neg text mean value: {neg_values_text.mean().item():.4f}")
        print(f"  Random neg text mean value: {random_values_text.mean().item():.4f}")
    
    ### --------- ###
    return (
        gram_loss,
        gram_volume,
        gram_volumeT,
        loss_d2a,
        loss_a2d,
        loss_dam
    )
