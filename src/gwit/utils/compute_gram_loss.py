import os
os.environ['HF_HOME'] = './cache/'
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import LayerNorm as LayerNorm
import torch.utils.checkpoint

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
    
class BERTLikeMultimodalEncoder(nn.Module):
    def __init__(self, clip_encoder, num_cross_layers=6):
        super().__init__()
        self.text_encoder = clip_encoder
        self.cross_layers = nn.ModuleList([
            CrossAttentionLayer(hidden_size=1024) 
            for _ in range(num_cross_layers)
        ])
        
    def forward(self, input_ids, attention_mask, condition_feats):
        # Get text features from CLIP
        text_features = self.text_encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        ).last_hidden_state
        
        # Apply cross-attention layers
        hidden_states = text_features
        for layer in self.cross_layers:
            hidden_states = layer(hidden_states, condition_feats)
            
        return hidden_states

# Trying to replicate the get_multimodal_forward_input_vision behaviour but for images
# by converting the inputs into a sequence of "tokens" that can be processed by the multimodal encoder's cross attention layer
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


def compute_gram_loss(latents, encoder_hidden_states, eeg_embedding, batch, feature_queue, controlnet, 
                      latents_proj, text_proj, controlnet_image_proj, temperature, accelerator, 
                      hidden_trans_image_multimodal, hidden_trans_eeg_multimodal, text_encoder):
    
    itm_head = Match_head(1024).to(accelerator.device)
    ### --------- ###
    # POOLING
    pooled_latents = latents.mean(dim=[2,3]) #should be [b,4]
    # With CLIP the eos token has a similar role to the CLS token in BERT, so we take it as the pooled representation
    pooled_captions = encoder_hidden_states[:,-1,:] #should be [b, 1024]
    #average across the spatial dimensions
    pooled_eeg_embedding = eeg_embedding.mean(dim=[2,3]) #should be [b,320]

    #Projections
    projected_latents = latents_proj(pooled_latents) #should be [b,128]
    projected_captions = text_proj(pooled_captions) #should be [b,128]
    projected_controlnet_image = controlnet_image_proj(pooled_eeg_embedding) #should be [b,128]

    # normalize
    projected_latents = F.normalize(projected_latents, dim=-1)
    projected_captions = F.normalize(projected_captions, dim=-1)
    projected_controlnet_image = F.normalize(projected_controlnet_image, dim=-1)
                
    # add embeddings to queue so we can compare current batch with more negative samples
    feature_queue["latents"].extend(projected_latents.detach().cpu())
    feature_queue["captions"].extend(projected_captions.detach().cpu())
    feature_queue["eeg"].extend(projected_controlnet_image.detach().cpu())

    # get samples from queue and stack it 
    gathered_latents = torch.stack(list(feature_queue["latents"]), dim=0).to(accelerator.device)
    print("gathered_latents shape: ", gathered_latents.shape)
    gathered_captions = torch.stack(list(feature_queue["captions"]), dim=0).to(accelerator.device)
    print("gathered_captions shape: ", gathered_captions.shape)
    gathered_eeg = torch.stack(list(feature_queue["eeg"]), dim=0).to(accelerator.device)
    print("gathered_eeg shape: ", gathered_eeg.shape)

    # combine gathered samples with current batch
    combined_latents = torch.cat([projected_latents, gathered_latents], dim=0)
    print("combined_latents shape: ", combined_latents.shape)
    combined_captions = torch.cat([projected_captions, gathered_captions], dim=0)
    print("combined_captions shape: ", combined_captions.shape)
    combined_eeg = torch.cat([projected_controlnet_image, gathered_eeg], dim=0)
    print("combined_eeg shape: ", combined_eeg.shape)

    # volume computation
    gram_volume = volume_computation3(projected_captions, combined_latents, combined_eeg) #volume_computation3(projected_captions, combined_latents, combined_eeg)
    gram_volume = gram_volume / temperature
    gram_volumeT = volume_computation3(combined_captions, projected_latents, projected_controlnet_image).T #volume_computation3(combined_captions, projected_latents, projected_controlnet_image).T
    gram_volumeT = gram_volumeT / temperature
                
    # Compute gram losses:
    batch_size = projected_captions.size(0)
    print("batch_size: ", batch_size)
    print("gram_volume shape: ", gram_volume.shape)
    print("gram_volumeT shape: ", gram_volumeT.shape)
    print("Diagonal values of gram_volume:", torch.diagonal(gram_volume[:batch_size, :batch_size]))
    print("Mean diagonal value:", torch.diagonal(gram_volume[:batch_size, :batch_size]).mean())
    print("Mean off-diagonal value:", (gram_volume[:batch_size, :batch_size].sum() - torch.diagonal(gram_volume[:batch_size, :batch_size]).sum()) / (batch_size * (batch_size - 1)))
                
    targets = torch.arange(batch_size).to(accelerator.device)
    loss_d2a = F.cross_entropy(-gram_volume, targets, label_smoothing=0.1)                
    loss_a2d = F.cross_entropy(-gram_volumeT, targets, label_smoothing=0.1) 

    gram_loss = (loss_d2a + loss_a2d) / 2
    #print("\n projected images and text cosine similarity",F.cosine_similarity(projected_latents, projected_captions, dim=-1).mean().detach().item())
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
    condition_feats_img = get_multimodal_forward_input_image(condition_feats_img, hidden_trans_image_multimodal, accelerator) 
    print("condition_feats_img shape: ", condition_feats_img.shape)
    condition_feats_eeg = eeg_embedding
    condition_feats_eeg = get_multimodal_forward_input_eeg(condition_feats_eeg, hidden_trans_eeg_multimodal)
    print("condition_feats_eeg shape: ", condition_feats_eeg.shape)

    condition_feats_img_eeg = torch.cat((condition_feats_img, condition_feats_eeg), dim=1)
    print("condition_feats_img_eeg shape: ", condition_feats_img_eeg.shape)
    feature_queue["condition_feats"].extend(condition_feats_img_eeg.detach().cpu())

    '''
    condition_feats_collate = all_gather_with_grad(condition_feats)
    '''
    condition_feats_collate = torch.stack(list(feature_queue["condition_feats"]), dim=0).to(accelerator.device)
    print("condition_feats_collate shape: ", condition_feats_collate.shape)
    
    input_ids = batch['input_ids']
    feature_queue["text_ids"].extend(input_ids.detach().cpu())
    print("input_ids shape: ", input_ids.shape)
    input_ids_collate = torch.stack(list(feature_queue["text_ids"]), dim=0).to(accelerator.device)
    print("input_ids_collate shape: ", input_ids_collate.shape)

    attention_mask = batch['attention_mask']
    feature_queue["attention_mask"].extend(attention_mask.detach().cpu())
    print("attention_mask shape: ", attention_mask.shape)
    attention_mask_collate = torch.stack(list(feature_queue["attention_mask"]), dim=0).to(accelerator.device)
    print("attention_mask_collate shape: ", attention_mask_collate.shape)

    # From original code
    with torch.no_grad():
        weights_t2cond = F.softmax(-(gram_volume), dim=1) + 1e-4
        print("weights_t2cond shape: ", weights_t2cond.shape)
        weights_t2cond[:, :batch_size].fill_diagonal_(0)

        weights_cond2t = F.softmax(-(gram_volumeT), dim=1) + 1e-4
        weights_cond2t[:, :batch_size].fill_diagonal_(0)

    condition_feats_neg = []
    for b in range(batch_size):
        # Limit the sampling to valid indices
        valid_range = min(weights_t2cond.shape[1], condition_feats_collate.shape[0])
        # Sample from the valid range only
        limited_weights = weights_t2cond[b, :valid_range]
        neg_idx = torch.multinomial(limited_weights, 1).item()
        condition_feats_neg.append(condition_feats_collate[neg_idx])
    condition_feats_neg = torch.stack(condition_feats_neg, dim=0)


    text_ids_neg = []
    text_atts_neg = []
    for b in range(batch_size):
        neg_idx = torch.multinomial(weights_cond2t[b, :batch_size], 1).item()
        text_ids_neg.append(input_ids_collate[neg_idx])
        text_atts_neg.append(attention_mask_collate[neg_idx])

    text_ids_neg = torch.stack(text_ids_neg, dim=0)
    text_atts_neg = torch.stack(text_atts_neg, dim=0)

    input_ids_1 = torch.cat((input_ids, input_ids, text_ids_neg),dim=0)
    attention_mask_1 = torch.cat((attention_mask, attention_mask, text_atts_neg),dim=0)
    
    condition_feats = torch.cat((condition_feats_img_eeg,condition_feats_neg,condition_feats_img_eeg),dim=0)

    #cross_attention = CrossAttentionLayer(hidden_size=1024).to(accelerator.device)
    multimodal_encoder = BERTLikeMultimodalEncoder(text_encoder).to(accelerator.device)
    output = multimodal_encoder(input_ids=input_ids_1, attention_mask=attention_mask_1, condition_feats=condition_feats)
    #cross_attended_output = cross_attention(output.last_hidden_state, condition_feats)


    batch_size = condition_feats_neg.shape[0]
    logits = itm_head(output[:,0])#.half())
    ground_truth = torch.zeros(batch_size*3).long().cuda()
    ground_truth[:batch_size] = 1
    loss_dam = F.cross_entropy(logits,ground_truth) #itm (dtm)
    print("loss_dam: ", loss_dam)
    #dam_loss = (0.1 * loss)
    
    print("Logging for dam loss:")
    print("Selected negative indices for conditions:", [torch.multinomial(weights_t2cond[b], 1).item() for b in range(batch_size)])
    print("Selected negative indices for text:", [torch.multinomial(weights_cond2t[b], 1).item() for b in range(batch_size)])
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
