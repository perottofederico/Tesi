import os
from pathlib import Path
import logging
os.environ['HF_HOME'] = './Tesi/cache/'

from PIL import Image
import sys
import math
import random
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils.torch_utils import is_compiled_module
from transformers import (
    CLIPVisionModel, CLIPFeatureExtractor,
    CLIPTextModel, CLIPTokenizer, CLIPProcessor
)
from torch.nn import LayerNorm as LayerNorm
from diffusers.optimization import get_scheduler
import transformers
from transformers import AutoTokenizer, PretrainedConfig
from diffusers.utils import check_min_version, is_wandb_available
from utils import make_train_dataset, train_collate_fn, make_eval_dataset, eval_collate_fn, parse_args
import diffusers
from diffusers import AutoencoderKL

from gram_utils import volume_computation3
from utils.distributed import concat_all_gather, all_gather_with_grad

from controlnet_conditioning_eeg import ControlNetEEGConditioningEmbedding
from pretrain_validation import evaluate_pretraining
from tsne_viz_CVPR import plot_tsne
from tsne_viz import plot_tsne_eeg, create_embedding_progression_gif

from tqdm import tqdm
if is_wandb_available():
    import wandb


logger = get_logger(__name__)
# GELU is only used in the Match_head, which is never called
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

# This is never called
class Match_head(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.activation = GELU()
        self.layernorm = LayerNorm(hidden_size, eps=1e-12)
        self.linear2 = nn.Linear(hidden_size, 2)
    def forward(self, cls_token):
        return self.linear2(self.layernorm(self.activation(self.linear1(cls_token))))

# This is never called
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

    def import_model_class_from_model_name_or_path(self, pretrained_model_name_or_path: str, revision: str):
        text_encoder_config = PretrainedConfig.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=revision,
        )
        model_class = text_encoder_config.architectures[0]

        if model_class == "CLIPTextModel":
            from transformers import CLIPTextModel

            return CLIPTextModel
        elif model_class == "RobertaSeriesModelWithTransformation":
            from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

            return RobertaSeriesModelWithTransformation
        else:
            raise ValueError(f"{model_class} is not supported.")

    def load_image_encoder(self, args):
        ## OPTION 1: VAE 
        #print("Loading VAE")
        #vae = AutoencoderKL.from_pretrained(
        #    args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        #)
        #self.vae = vae
        #self.vae.requires_grad_(False)
        #print("Done!")
        
        ## OPTION 2: CLIP base-32
        #print("Loading CLIP image encoder")
        #self.vae = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        #self.vae.requires_grad_(False)
        #print("Weights are frozen")
        #print("Done!")

        ## OPTION 3: CLIP ViT-H-14 (laion)
        #print("Loading Laion CLIP ViT-H-14 image encoder")
        self.vae = CLIPVisionModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.vae.requires_grad_(False)
        #print("Done!")
    
    def load_text_encoder(self, args):
        ## OPTION 1: SD Text Encoder (CLIP ViT-H-14)
        print("Loading Tokenizer")
        if args.tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
        elif args.pretrained_model_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=args.revision,
                use_fast=False,
            )
        # import correct text encoder class
        text_encoder_cls = self.import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
        print("Loading Text Encoder")
        self.text_encoder = text_encoder_cls.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
        )
        self.text_encoder.requires_grad_(False)
        #print("Weights are frozen")
        print("Done!")

        ## OPTION 2: CLIP base-32
        #from transformers import CLIPTextModel
        #self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        #self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        #self.text_encoder.requires_grad_(False)

        ## OPTION 3: CLIP ViT-H-14 (laion)
        # Is it the same as option 1?
        #print("Loading Laion CLIP ViT-H-14 text encoder")
        #self.tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        #self.text_encoder = CLIPTextModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        #self.text_encoder.requires_grad_(True)
        #print("Done!")

    def load_eeg_encoder(self, args):
        """
        Load the EEG encoder based on the specified type in args.
        """
        if args.eeg_encoder_name == 'gwit':
            self.load_gwit_eeg_encoder()
        elif args.eeg_encoder_name == 'lstm':
            self.load_LSTM_eeg_encoder()
        else:
            raise ValueError(f"Unknown EEG encoder type: {args.eeg_encoder_type}")

    def load_gwit_eeg_encoder(self):
        self.eeg_encoder = ControlNetEEGConditioningEmbedding(
            conditioning_embedding_channels= 320, #block_out_channels[0],
            n_subjects= 7 # n_subjects #7 #24 TODO add this changing with the dataset
            # block_out_channels=conditioning_embedding_out_channels, # default value is  (16, 32, 96, 256)   I LORO
            # conditioning_channels=conditioning_channels, default value is 128
        )
        self.eeg_encoder.requires_grad_(True)
    
    def load_LSTM_eeg_encoder(self):
        ## OPTION 2: EEGFeatNet
        print("Loading EEG encoder")
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        base_dir = os.path.dirname(current_dir)
        path_to_append = base_dir+f"/EEGStyleGAN-ADA/EEG2Feat/Triplet_LSTM/CVPR40" if "CVPR" in args.dataset_name else base_dir+f"/EEGStyleGAN-ADA/EEG2Feat/Triplet_LSTM/Thoughtviz"
        sys.path.append(path_to_append)
        from network import EEGFeatNet

        #Try loading the EEGFeatNet model with its own weights
        model = EEGFeatNet(n_features=128, projection_dim=128, num_layers=4).to("cuda") if "CVPR" in args.dataset_name else  \
                EEGFeatNet(n_classes=10, in_channels=14, n_features=128, projection_dim=128, num_layers=4).to("cuda")
        model = torch.nn.DataParallel(model).to("cuda")

        ckpt_path = base_dir+"/EEGStyleGAN-ADA/EEG2Feat/Triplet_LSTM/CVPR40/EXPERIMENT_29/bestckpt/eegfeat_all_0.9665178571428571.pth" if "CVPR" in args.dataset_name \
            else base_dir+'/EEGStyleGAN-ADA/EEG2Feat/Triplet_LSTM/Thoughtviz/EXPERIMENT_1/bestckpt/eegfeat_all_0.7212357954545454.pth' 
        model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
        self.eeg_encoder = model
        self.eeg_encoder.requires_grad_(True)
        print("Done!")

    def pool_text_for_contra(self, feature):
        return feature.pooler_output 
    
    def pool_image_for_contra(self, feature):
        return feature.pooler_output #CLIP CLS pooled, main option
        #return feature.last_hidden_state.mean(dim=1) #Alternative pooling for CLIP
        #return feature.mean(dim=[2,3]) #VAE
        #return feature[:,0,:]
    
    def pool_eeg_for_contra(self, feature, encoder_name):
        #if self.config.eeg_encoder_type.startswith('controlnet'):
        #    feature = feature.mean(dim=-1) #Global average pooling
        #else: # for when I add eegFeatNet
        #    raise NotImplementedError
        #feature = torch.mean(feature, dim=1)
        
        #return feature.mean(dim=-1) #Global average pooling, for gwit vec encoder
        if encoder_name == 'gwit':
            #return feature.mean(dim=[2,3]) # gwit normal
            return feature.mean(dim=-1) # gwit vector type
        else : # for eegFeatNet
            return feature
        
class GRAM(MMGeneralModule):
    def __init__(self, args):
        super().__init__()

        self.load_image_encoder(args)
        self.load_text_encoder(args)
        self.load_eeg_encoder(args)

        contra_dim = 512 # if args.eeg_encoder_name == 'gwit' else 128 # TODO consider changing this for lstm especially
        self.multimodal_dim = 1024 #text encoder hidden size
        self.image_dim = 1280 #Clip=1280, VAE = 
        self.eeg_dim = 2560 if args.eeg_encoder_name=='gwit' else 128 #TODO try gwit vector type dim =2560, 320 for normal

        self.contra_head_t = Contra_head(self.multimodal_dim, contra_dim)
        self.contra_head_i = Contra_head(self.image_dim, contra_dim)
        self.contra_head_e = Contra_head(self.eeg_dim, contra_dim)
        self.contra_temp = nn.Parameter(torch.tensor(0.07))

    def batch_get(self, batch, key):
        if key in batch:
            return batch[key]
        # I already have input_ids and attention_mask in the batch
        #elif key == 'caption_tokens':
        #    caption_tokens = self.text_encoder.tokenizer(batch.raw_captions,
        #                                            padding="max_length",
        #                                            truncation=True,
        #                                            max_length=self.max_caption_len,
        #                                            return_tensors="pt").to(accelerator.device)
        #    batch[key] = caption_tokens
        elif key =='input_ids':
            input_ids = batch['input_ids']
        elif key == 'attention_mask':
            attention_mask = batch['attention_mask']
        elif key == 'caption_output':
            input_ids = self.batch_get(batch, 'input_ids')
            attention_mask = self.batch_get(batch, 'attention_mask')
            caption_output = self.text_encoder(input_ids = input_ids,
                                            attention_mask = attention_mask)#.last_hidden_state
            batch[key] = caption_output
        elif key == 'feat_t':
            caption_output = self.batch_get(batch, 'caption_output')
            caption_output_pooled = self.pool_text_for_contra(caption_output)
            feat_t = self.contra_head_t(caption_output_pooled)
            feat_t = F.normalize(feat_t,dim=-1)
            batch[key] = feat_t

       
        elif key == 'image_output':
            #clip
            image_output = self.vae(batch["pixel_values"].to(dtype=torch.float32))#.last_hidden_state
            batch[key] = image_output
        elif key == 'feat_i':
            image_output = self.batch_get(batch, 'image_output')
            image_output_pooled = self.pool_image_for_contra(image_output)
            feat_i = self.contra_head_i(image_output_pooled)
            feat_i = F.normalize(feat_i,dim=-1)
            batch[key] = feat_i


        elif key == 'subjects':
            subjects = batch['eeg_subjects']
            batch[key] = subjects
        elif key == 'eeg_output':
            if args.eeg_encoder_name == 'gwit':
                subjects = self.batch_get(batch, 'subjects')
                eeg_output = self.eeg_encoder(batch['conditioning_pixel_values'], subjects, return_vector = True)
                batch[key] = eeg_output
            elif args.eeg_encoder_name == 'lstm':
                eeg = (batch['conditioning_pixel_values']).permute(0,2,1)
                eeg_output = self.eeg_encoder(eeg)
                batch[key] = eeg_output[1]
        elif key == 'feat_e':
            eeg_output = self.batch_get(batch, 'eeg_output')
            eeg_output_pooled = self.pool_eeg_for_contra(eeg_output, args.eeg_encoder_name)
            feat_e = self.contra_head_e(eeg_output_pooled)
            feat_e = F.normalize(feat_e,dim=-1)
            batch[key] = feat_e

        return batch[key]

    def forward(self, batch):
        # feat_t
        feat_t = self.batch_get(batch, 'feat_t')
        feat_t_all = concat_all_gather(feat_t)
        
        # feat_i
        feat_i = self.batch_get(batch, 'feat_i')
        feat_i_all = concat_all_gather(feat_i)

        # feat_e
        feat_e = self.batch_get(batch, 'feat_e')
        feat_e_all = concat_all_gather(feat_e)
        
        volume = volume_computation3(feat_t,feat_i_all,feat_e_all)
        #volume = volume_computation3(feat_i, feat_t_all,feat_e_all) # img anchor
        volume = volume / self.contra_temp
        volumeT = volume_computation3(feat_t_all,feat_i,feat_e).T
        #volumeT = volume_computation3(feat_i_all, feat_t, feat_e).T # img anchor
        volumeT = volumeT / self.contra_temp

        rank = 0 #accelerator.process_index TODO generalize this for multiple GPUs
        bs = feat_t.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(volume.device)
        loss = (
                    F.cross_entropy(-volume, targets, label_smoothing=0.1) #d2a
                    + F.cross_entropy(-volumeT, targets, label_smoothing=0.1) #a2d
        ) / 2
        return loss
    
def volume_computation(anchor, *inputs):
    """
    General function to compute volume for contrastive learning loss functions.
    Compute the volume metric for each vector in anchor batch and all the other modalities listed in *inputs.

    Args:
    - anchor (torch.Tensor): Tensor of shape (batch_size1, dim)
    - *inputs (torch.Tensor): Variable number of tensors of shape (batch_size2, dim)

    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
    """
    batch_size1 = anchor.shape[0]
    batch_size2 = inputs[0].shape[0]

    # Compute pairwise dot products for language with itself
    aa = torch.einsum('bi,bi->b', anchor, anchor).unsqueeze(1).expand(-1, batch_size2)

    # Compute pairwise dot products for language with each input
    l_inputs = [anchor @ input.T for input in inputs]

    # Compute pairwise dot products for each input with themselves and with each other
    input_dot_products = []
    for i, input1 in enumerate(inputs):
        row = []
        for j, input2 in enumerate(inputs):
            dot_product = torch.einsum('bi,bi->b', input1, input2).unsqueeze(0).expand(batch_size1, -1)
            row.append(dot_product)
        input_dot_products.append(row)

    # Stack the results to form the Gram matrix for each pair
    G = torch.stack([
        torch.stack([aa] + l_inputs, dim=-1),
        *[torch.stack([l_inputs[i]] + input_dot_products[i], dim=-1) for i in range(len(inputs))]
    ], dim=-2)

    # Compute the determinant for each Gram matrix
    gram_det = torch.det(G.float())

    # Compute the square root of the absolute value of the determinants
    res = torch.sqrt(torch.abs(gram_det))
    return res


def main(args):
    
    # Preamble
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            
    # Some housekeeeping
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
        # `accelerate` 0.16.0 will have better support for customized saving
   
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    
    # 3) Create GRAM model 
    gram_model = GRAM(args)
    gram_model.train()
    
    # 4) Load train and evaluation datasets
    print("Loading Dataset")
    train_dataset = make_train_dataset(args, gram_model.tokenizer, accelerator)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=train_collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    eval_dataset = make_eval_dataset(args, gram_model.tokenizer, accelerator)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        shuffle=False,
        collate_fn=eval_collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers
    )

    # 5) optimizer
    params_to_optimize = list(gram_model.parameters())
    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Some more housekeeeping
    # Scheduler creation and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    gram_model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        gram_model, optimizer, train_dataloader, eval_dataloader ,lr_scheduler
    )

    # Move text, img and eeg encoder to device and cast to weight_dtype
    gram_model.to(accelerator.device, dtype=torch.float32)
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)
    
    # Log the model configuration
    logger.info("Model configuration:")
    logger.info(f"Dataset: {args.dataset_name}")
    logger.info(f"Contrastive dim: {gram_model.contra_head_t.linear.out_features}")
    logger.info(f"Image/Text encoder frozen: {not any(p.requires_grad for p in gram_model.vae.parameters())}")
    logger.info(f"Batch size per device: {args.train_batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Subjects: {'ALL' if args.subject_num==0 else args.subject_num}")

    logger.info("Running evaluation step before training")
    evaluate_pretraining(gram_model, args, 0, accelerator, eval_dataset, eval_dataloader)
    plot_tsne_eeg(gram_model, eval_dataloader, args, step=0)
    gram_model.train()

    # training loop
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    initial_global_step = 0
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(gram_model): #TODO include other parts
                loss = gram_model(batch)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = gram_model.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        #save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        #accelerator.save_state(save_path)
                        logger.info(f"Skipped checkpoint saving to savepath") #TODO replace with actual path

                    # Validation step
                    if global_step % args.validation_steps == 0:
                        evaluate_pretraining(gram_model, args, global_step, accelerator, eval_dataset, eval_dataloader)
                        plot_tsne_eeg(gram_model, eval_dataloader, args, global_step)
                        gram_model.train()
            logs = {
                "loss": loss.detach().item(),
                "contra_temp": gram_model.contra_temp.detach().item(),
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        create_embedding_progression_gif(f"{args.output_dir}")
        gram_model = unwrap_model(gram_model)
        eeg = gram_model.eeg_encoder
        save_path = os.path.join(args.output_dir, "final_model")
        accelerator.save_model(gram_model, os.path.join(save_path, "gram"))
        accelerator.save_model(eeg, os.path.join(save_path, "eeg_encoder"))
        
    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)