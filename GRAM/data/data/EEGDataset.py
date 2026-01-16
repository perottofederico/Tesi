import os
import random
from toolz.sandbox import unzip

import sys
import pickle
import multiprocessing as _mp
#try:
#    _mp.set_start_method('spawn', force=True)
#except RuntimeError:
#    # already set
#    pass

import numpy as np
import hashlib
import torch.distributed as dist
import torch
import torch.nn.functional as F
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

        print("Loading Tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
                "stabilityai/stable-diffusion-2-1-base",
                subfolder="tokenizer",
                #revision=args.revision,
                use_fast=False,
        )
        print("Done!")
        
        if(d_cfg["split"] == "train"):
            self.data = make_train_dataset(self, d_cfg, tokenizer)
        else:
            self.data = make_eval_dataset(self, d_cfg, tokenizer)

    
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
   
        return image, id_, id_txt, vision_pixels, conditioning_pixel_values, raw_captions, subject, label
        #return self.data[index]
        
def make_train_dataset(self, d_cfg, tokenizer):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # If the train data is a path, we assume it is a local dataset.

        # Downloading and loading a dataset from the hub.
        if d_cfg["good_captions"]:
            LOGGER.info("Loading training dataset with good captions")
            dataset = load_dataset(
                "perottofederico/EEGCVPR40_good_captions",
                split="train"
            )
        else:
            LOGGER.info("Loading training dataset with simple captions")
            dataset = load_dataset(
                "luigi-s/EEG_Image_CVPR_ALL_subj", #args.dataset_name,
                split="train"
            )

        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        if d_cfg.subject_num != []:
            if isinstance(d_cfg.subject_num, list):
                dataset = dataset.filter(lambda x: x['subject'] not in d_cfg.subject_num)
            else:
                dataset = dataset.filter(lambda x: x['subject'] != d_cfg.subject_num)
        
        if d_cfg.classes_to_remove != []:
            if isinstance(d_cfg.classes_to_remove, list):
                dataset = dataset.filter(lambda x: x['label'] not in d_cfg.classes_to_remove)
            else:
                dataset = dataset.filter(lambda x: x['label'] != d_cfg.classes_to_remove)

        column_names = dataset.column_names

        # 6. Get the column names for input/target.
        if d_cfg.image_column is None:
            image_column = column_names[0]
            LOGGER.info(f"image column defaulting to {image_column}")
        else:
            image_column = d_cfg.image_column
            if image_column not in column_names:
                raise ValueError(
                    f"`--image_column` value '{d_cfg.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )
        if d_cfg.caption_column is None:
            caption_column = column_names[1]
            LOGGER.info(f"caption column defaulting to {caption_column}")
        else:
            caption_column = d_cfg.caption_column
            if caption_column not in column_names:
                raise ValueError(
                    f"`--caption_column` value '{d_cfg.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )
        if d_cfg.conditioning_image_column is None:
            conditioning_image_column = column_names[2]
            LOGGER.info(f"conditioning image column defaulting to {conditioning_image_column}")
        else:
            conditioning_image_column = d_cfg.conditioning_image_column
            if conditioning_image_column not in column_names:
                raise ValueError(
                    f"`--conditioning_image_column` value '{d_cfg.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )
        
        def tokenize_captions(examples):
            # iterate over examples[caption_column] an add half of the captions as empty strings
            # and the other half as the original caption in the captions list
            captions = []
            for caption in examples[caption_column]:
                if isinstance(caption, str):
                    captions.append(caption) 
                elif isinstance(caption, (list, np.ndarray)):
                    # take a random caption if there are multiple
                    i+=1
                    captions.append(random.choice(caption))
                else:
                    raise ValueError(
                        f"Caption column `{caption_column}` should contain either strings or lists of strings."
                    )
            # Tokenize the captions
            inputs = tokenizer(
                captions, padding=True, truncation=True, return_tensors="pt"
            )
            # Return the tokenized captions (shape [1,77])
            print("=========",i)
            return inputs.input_ids, inputs.attention_mask

        def get_caption_from_classifier(eeg, labels):
            # Get the current file path and directory
            current_file_path = os.path.abspath(__file__)
            current_dir = os.path.dirname(current_file_path)

            # Go up two levels from the current directory
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            path_to_append = base_dir+f"/Tesi/src/EEGStyleGAN-ADA/EEG2Feat/Triplet_LSTM/CVPR40" if "CVPR" in self.dataset_name else base_dir+f"/Tesi/src/EEGStyleGAN-ADA/EEG2Feat/Triplet_LSTM/Thoughtviz"
            sys.path.append(path_to_append)
            from network import EEGFeatNet
            if "CVPR" in self.dataset_name:
                from dataset_EEG.name_map_ID import id_to_caption
            else:
                from dataset_EEG.name_map_ID import id_to_caption_TVIZ as id_to_caption
            model = EEGFeatNet(n_features=128, projection_dim=128, num_layers=4).to("cuda") if "CVPR" in self.dataset_name else  \
                    EEGFeatNet(n_classes=10, in_channels=14, n_features=128, projection_dim=128, num_layers=4).to("cuda")
            model = torch.nn.DataParallel(model).to("cuda")

            # Load the model from the file
            pkl_path = base_dir+'/Tesi/src/gwit/dataset_EEG/knn_model.pkl' if "CVPR" in self.dataset_name else base_dir+'/Tesi/src/gwit/dataset_EEG/knn_model_TVIZ.pkl'
            with open(pkl_path, 'rb') as f:
                knn_cv = pickle.load(f)
            ckpt_path = base_dir+"/Tesi/src/EEGStyleGAN-ADA/EEG2Feat/Triplet_LSTM/CVPR40/EXPERIMENT_29/bestckpt/eegfeat_all_0.9665178571428571.pth" if "CVPR" in self.dataset_name \
                else base_dir+'/Tesi/src/EEGStyleGAN-ADA/EEG2Feat/Triplet_LSTM/Thoughtviz/EXPERIMENT_1/bestckpt/eegfeat_all_0.7212357954545454.pth' 
            model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])

            eeg =  torch.stack(eeg) if "CVPR" in self.dataset_name else torch.stack([torch.tensor(eeg_e) for eeg_e in eeg]) # stack all the eegs
            x_proj = model(eeg.view(-1,eeg.shape[2],eeg.shape[1]).to("cuda")) # reshape the eegs and pass them to the EEGFeatNet model
            labels = [torch.tensor(l) if not isinstance(l, torch.Tensor) else l for l in labels] # convert the labels to tensors (if they aren' already)
            # Predict the labels
            predicted_labels = knn_cv.predict(x_proj.cpu().detach().numpy())
            captions = ["image of " + id_to_caption[label] for label in predicted_labels] # add "image of" to the labels
            return captions
        
        def get_good_captions(images):
            captions = []
            for i, image in enumerate(images):
                inputs = self.blip_processor(images=image, return_tensors="pt").to("cuda", torch.float16)
                out = self.blip_model.generate(**inputs, max_new_tokens=30, num_beams=4, min_length=5)
                caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
                captions.append(caption)
            return captions

        def preprocess_train(train_split):
            image_transforms = transforms.Compose(
                [
                    transforms.Resize(d_cfg.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(d_cfg.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

            images = [image.convert("RGB") for image in train_split[image_column]] #Convert each pixel of each image in image_column to 3 8 bit values (via PIL)
            
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
            conditioning_images = [torch.tensor(image) for image in train_split[conditioning_image_column]] # transform all the conditioning images (eegs) to tensors

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


def make_eval_dataset(self, d_cfg, tokenizer):

        if d_cfg["good_captions"]:
            LOGGER.info("Loading eval dataset with good captions")
            dataset = load_dataset(
                "perottofederico/EEGCVPR40_good_captions",
                split="test" # I might have swapped test and validation splits when creating the dataset ops
            )
        else:
            LOGGER.info("Loading eval dataset with simple captions")
            dataset = load_dataset(
                "luigi-s/EEG_Image_CVPR_ALL_subj",
                split="validation"
            )

        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        if d_cfg.subject_num != 0:
            dataset = dataset.filter(lambda x: x['subject'] == d_cfg.subject_num)

        # in eval dataset, if we have "classes_to_keep" we keep only those classes
        if d_cfg.classes_to_keep != []:
            if isinstance(d_cfg.classes_to_keep, list):
                dataset = dataset.filter(lambda x: x['label'] in d_cfg.classes_to_keep)
            else:
                dataset = dataset.filter(lambda x: x['label'] == d_cfg.classes_to_keep)

        column_names = dataset.column_names

        # 6. Get the column names for input/target.
        if d_cfg.image_column is None:
            image_column = column_names[0]
            LOGGER.info(f"image column defaulting to {image_column}")
        else:
            image_column = d_cfg.image_column
            if image_column not in column_names:
                raise ValueError(
                    f"`--image_column` value '{d_cfg.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )

        if d_cfg.caption_column is None:
            caption_column = column_names[1]
            LOGGER.info(f"caption column defaulting to {caption_column}")
        else:
            caption_column = d_cfg.caption_column
            if caption_column not in column_names:
                raise ValueError(
                    f"`--caption_column` value '{d_cfg.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )

        if d_cfg.conditioning_image_column is None:
            conditioning_image_column = column_names[2]
            LOGGER.info(f"conditioning image column defaulting to {conditioning_image_column}")
        else:
            conditioning_image_column = d_cfg.conditioning_image_column
            if conditioning_image_column not in column_names:
                raise ValueError(
                    f"`--conditioning_image_column` value '{d_cfg.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )

        def tokenize_captions(examples):
            # iterate over examples[caption_column] an add half of the captions as empty strings
            # and the other half as the original caption in the captions list
            captions = []
            for caption in examples[caption_column]:
                if isinstance(caption, str):
                    captions.append(caption)
                elif isinstance(caption, (list, np.ndarray)):
                    # take a random caption if there are multiple
                    captions.append(random.choice(caption))
                else:
                    raise ValueError(
                        f"Caption column `{caption_column}` should contain either strings or lists of strings."
                    )
            # Tokenize the captions
            inputs = tokenizer(
                captions, padding=True, truncation=True, return_tensors="pt"
            )
            # Return the tokenized captions (shape [1,77])
            return inputs.input_ids, inputs.attention_mask

        def get_caption_from_classifier(eeg, labels):
            # Get the current file path and directory
            current_file_path = os.path.abspath(__file__)
            current_dir = os.path.dirname(current_file_path)

            # Go up two levels from the current directory
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            print(base_dir, "aaaaaaa")
            path_to_append = base_dir+f"/Tesi/src/EEGStyleGAN-ADA/EEG2Feat/Triplet_LSTM/CVPR40" if "CVPR" in self.dataset_name else base_dir+f"/Tesi/src/EEGStyleGAN-ADA/EEG2Feat/Triplet_LSTM/Thoughtviz"
            print(path_to_append)
            sys.path.append(path_to_append)
            from network import EEGFeatNet
            if "CVPR" in self.dataset_name:
                from dataset_EEG.name_map_ID import id_to_caption
                print("Ids to caption: ", id_to_caption)
            else:
                from dataset_EEG.name_map_ID import id_to_caption_TVIZ as id_to_caption
                print(id_to_caption)
            model = EEGFeatNet(n_features=128, projection_dim=128, num_layers=4).to("cuda") if "CVPR" in self.dataset_name else  \
                    EEGFeatNet(n_classes=10, in_channels=14, n_features=128, projection_dim=128, num_layers=4).to("cuda")
            model = torch.nn.DataParallel(model).to("cuda")

            # Load the model from the file
            pkl_path = base_dir+'/Tesi/src/gwit/dataset_EEG/knn_model.pkl' if "CVPR" in self.dataset_name else base_dir+'/Tesi/src/gwit/dataset_EEG/knn_model_TVIZ.pkl'
            with open(pkl_path, 'rb') as f:
                knn_cv = pickle.load(f)
            ckpt_path = base_dir+"/Tesi/src/EEGStyleGAN-ADA/EEG2Feat/Triplet_LSTM/CVPR40/EXPERIMENT_29/bestckpt/eegfeat_all_0.9665178571428571.pth" if "CVPR" in self.dataset_name \
                else base_dir+'/Tesi/src/EEGStyleGAN-ADA/EEG2Feat/Triplet_LSTM/Thoughtviz/EXPERIMENT_1/bestckpt/eegfeat_all_0.7212357954545454.pth' 
            model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
            
            eeg =  torch.stack(eeg) if "CVPR" in self.dataset_name else torch.stack([torch.tensor(eeg_e) for eeg_e in eeg]) # stack all the eegs
            x_proj = model(eeg.view(-1,eeg.shape[2],eeg.shape[1]).to("cuda")) # reshape the eegs and pass them to the EEGFeatNet model
            labels = [torch.tensor(l) if not isinstance(l, torch.Tensor) else l for l in labels] # convert the labels to tensors (if they aren' already)
            # Predict the labels
            predicted_labels = knn_cv.predict(x_proj.cpu().detach().numpy())
            captions = ["image of " + id_to_caption[label] for label in predicted_labels] # add "image of" to the labels
            return captions

        def get_good_captions(images):
            #TESTING GOOD CAPTIONS
            captions = []
            for i, image in enumerate(images):
                inputs = self.blip_processor(images=image, return_tensors="pt").to("cuda", torch.float16)
                out = self.blip_model.generate(**inputs, max_new_tokens=30, num_beams=4, min_length=5)
                caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
                captions.append(caption)
            return captions
        
        def preprocess_train(train_split):
            
            image_transforms = transforms.Compose(
                [
                    transforms.Resize(d_cfg.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(d_cfg.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

            images = [image.convert("RGB") for image in train_split[image_column]] #Convert each pixel of each image in image_column to 3 8 bit values (via PIL)
            
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
            conditioning_images = [torch.tensor(image) for image in train_split[conditioning_image_column]] # transform all the conditioning images (eegs) to tensors

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
            'labels'
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
