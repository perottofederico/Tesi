import os
import random
from toolz.sandbox import unzip
import sys
import pickle
import numpy as np
import hashlib
import torch.distributed as dist
import torch
from torch.utils.data import Dataset
import torch.utils.checkpoint
from datasets import load_dataset
from torchvision import transforms
from utils.logger import LOGGER
from transformers import AutoTokenizer, PretrainedConfig, BertTokenizer
from .name_map_ID import name_map, folder_label_map, id_to_caption

class EEGDataset(Dataset):

    def __init__(self, d_cfg, args):
        self.name = d_cfg['name']
        self.dataset_name = d_cfg['name']
        self.training = d_cfg.training
        self.folder_label_map = folder_label_map

        self.worker_init_fn =  None
        self.use_sampler = True
        self.collate_fn = train_collate_fn
        
        if(d_cfg["split"] == "train"):
            self.data = make_train_dataset(self, d_cfg)
        else:
            self.data = make_eval_dataset(self, d_cfg)

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        image = self.data[index]["image"]
        id_ = self.data[index]["image_ids"]
        raw_captions = self.data[index]["caption"]
        raw_captions = raw_captions[0] if isinstance(raw_captions, list) else raw_captions
        
        num_samples = len(raw_captions) if isinstance(raw_captions, list) else 1
        id_txt = [id_] * num_samples
        
        vision_pixels = self.data[index]["pixel_values"]
        
        conditioning_pixel_values = self.data[index]["conditioning_pixel_values"]
        
        if "subject" in self.data[index]:
            subject = torch.as_tensor(self.data[index]["subject"])
        
        if "label" in self.data[index]:
            label = self.data[index]["label"]
            
        #pre computed features
        text_last_hidden_states = self.data[index]["text_last_hidden_state"]
        text_pooler_output = self.data[index]["text_pooler_output"]
        text_features = self.data[index]["text_features"]
        
        image_last_hidden_states = self.data[index]["image_last_hidden_state"]
        image_pooler_output = self.data[index]["image_pooler_output"]
        image_features = self.data[index]["image_features"]
   
        return (
            image, 
            id_, 
            id_txt, 
            vision_pixels, 
            conditioning_pixel_values, 
            raw_captions, 
            subject, 
            label,
            text_last_hidden_states,
            text_pooler_output,
            text_features,
            image_last_hidden_states,
            image_pooler_output,
            image_features
        )
        #return self.data[index]
        
def make_train_dataset(self, d_cfg):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # If the train data is a path, we assume it is a local dataset.

        # Downloading and loading a dataset from the hub.
        if d_cfg["good_captions"]:
            LOGGER.info("Loading training dataset with good captions")
            dataset = load_dataset(
                "perottofederico/EEGCVPR40_GoodCaptions",
                split="train"
            )
        else:
            LOGGER.info("Loading training dataset with simple captions")
            dataset = load_dataset(
                "luigi-s/EEG_Image_CVPR_ALL_subj", #args.dataset_name,
                split="train"
            )

        # Preprocessing the dataset
        def preprocess_train(train_split):
            image_transforms = transforms.Compose(
                [
                    transforms.Resize(d_cfg.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(d_cfg.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

            images = [image.convert("RGB") for image in train_split[d_cfg.image_column]] #Convert each pixel of each image in image_column to 3 8 bit values (via PIL)
            
             # add image ids (same id for idental images)
            image_ids = []
            for image in images:
                # Convert to numpy array before any transforms
                img_array = np.array(image.resize((64, 64)))  # Resize for consistent hashing
                # Hash the raw pixel data
                image_id = hashlib.md5(img_array.tobytes()).hexdigest()[:12]
                image_ids.append(image_id)
            
            train_split["image_ids"] = image_ids
            
            images = [image_transforms(image) for image in images] #Apply the transforms to each image
            
            #EEG
            conditioning_images = [torch.tensor(image) for image in train_split[d_cfg.conditioning_image_column]] # transform all the conditioning images (eegs) to tensors

            train_split["pixel_values"] = images # Add the pixel values to the train_split 
            train_split["conditioning_pixel_values"] = conditioning_images # Add the conditioning pixel values to the train_split 
            
            return train_split

        if dist.get_rank() == 0:
            train_dataset = dataset.map(preprocess_train, batched=True) #dataset["train"].with_transform(preprocess_train)
            train_dataset = train_dataset.with_format("torch")
        
        #print("Dataset Features: ", train_dataset.features)
        #print("Images: ", train_dataset[0]['pixel_values'].shape)
        #print("EEG: ", train_dataset[0]['conditioning_pixel_values'].shape)
        #print("Text: ", train_dataset[0]['input_ids'].shape)
        #print("Attention mask: ", train_dataset[0]['attention_mask'].shape)
        print("Caption example: ", train_dataset[0]['caption'])
        print("Total samples in train dataset: ", len(train_dataset))
        # print unique labels in the dataset
        return train_dataset


def make_eval_dataset(self, d_cfg):

        if d_cfg["good_captions"]:
            LOGGER.info("Loading eval dataset with good captions")
            dataset = load_dataset(
                "perottofederico/EEGCVPR40_GoodCaptions",
                split="validation" 
            )
        else:
            LOGGER.info("Loading eval dataset with simple captions")
            dataset = load_dataset(
                "luigi-s/EEG_Image_CVPR_ALL_subj",
                split="validation"
            )

        # Preprocessing the datasets
        def preprocess_train(train_split):
            
            image_transforms = transforms.Compose(
                [
                    transforms.Resize(d_cfg.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(d_cfg.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

            images = [image.convert("RGB") for image in train_split[d_cfg.image_column]] #Convert each pixel of each image in image_column to 3 8 bit values (via PIL)
            
            # add image ids (same id for idental images)
            image_ids = []
            for image in images:
                img_array = np.array(image.resize((64, 64))) 
                image_id = hashlib.md5(img_array.tobytes()).hexdigest()[:12]
                image_ids.append(image_id)
            
            train_split["image_ids"] = image_ids
            
            #Apply the transforms to each image
            images = [image_transforms(image) for image in images] 
            #EEG
            conditioning_images = [torch.tensor(image) for image in train_split[d_cfg.conditioning_image_column]] # transform all the conditioning images (eegs) to tensors

            train_split["pixel_values"] = images # Add the pixel values to the train_split 
            train_split["conditioning_pixel_values"] = conditioning_images # Add the conditioning pixel values to the train_split 

            #if d_cfg.caption_from_classifier:
            #    print("replacing captions with classifier captions")
            #    eeg_key = "conditioning_pixel_values" if "CVPR" in self.dataset_name else "eeg_no_resample"
            #    train_split[caption_column] = get_caption_from_classifier(train_split[eeg_key], train_split["label"]) # pass to the helper function the eegs (in tensor form) and the labels

            #train_split[caption_column] = get_good_captions(train_split[image_column])

            #train_split["input_ids"], train_split["attention_mask"] = tokenize_captions(train_split) # Tokenize the captions we have generated
            return train_split


        if dist.get_rank() == 0:
            eval_dataset = dataset.map(preprocess_train, batched=True)
            eval_dataset = eval_dataset.with_format("torch")
        #print("Dataset Features: ", eval_dataset.features)
        #print("Images: ", eval_dataset[0]['pixel_values'].shape)
        #print("EEG: ", eval_dataset[0]['conditioning_pixel_values'].shape)
        #print("Text: ", eval_dataset[0]['input_ids'].shape)
        #print("Attention mask: ", eval_dataset[0]['attention_mask'].shape)
        print("Caption example: ", eval_dataset[0]['caption'])
        print("Total samples in eval dataset: ", len(eval_dataset))
        return eval_dataset


def train_collate_fn(examples):
    batch = {}
    all_data = map(list, unzip(examples))
    
    keys = ['images',
            'ids', 
            'ids_txt',
            'vision_pixels',  
            "conditioning_pixel_values",
            'raw_captions', 
            'eeg_subjects', 
            'labels',
            'text_last_hidden_states',
            'text_pooler_output',
            'text_features',
            'image_last_hidden_states',
            'image_pooler_output',
            'image_features'
            ]

    for key, data in zip(keys, all_data):
        if data[0] is None:
            continue 
        elif isinstance(data[0], torch.Tensor):
            batch[key] = torch.stack(data, dim=0)#.float()
        else:
            batch[key] = data
    return batch

    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    #conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()
    #input_ids = torch.stack([example["input_ids"] for example in examples])
    #attention_mask = torch.stack([example["attention_mask"] for example in examples])
    #attention_mask = attention_mask.to(memory_format=torch.contiguous_format).float()
    
    # if "ALL" in args.dataset_name:
        # print([example["subject"] for example in examples])
        # Convert each integer subject to a tensor before stacking
    subjects = torch.stack([torch.as_tensor(example["subject"]) for example in examples])

    raw_captions = [example["caption"] for example in examples]
    ids = [example["label"] for example in examples]
    ids_txt = [ids]#*len(raw_captions)
    return {
        "vision_pixels": pixel_values,#"pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        #"input_ids": input_ids,
        #"attention_mask": attention_mask,
        "eeg_subjects": subjects, # if "ALL" in args.dataset_name else torch.tensor([4]*input_ids.shape[0]),
        "raw_captions": raw_captions,
        "ids": ids,
        "ids_txt": ids_txt
    }
