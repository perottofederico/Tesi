import hashlib
from typing import List, Optional
from toolz.sandbox import unzip
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
from torchvision import transforms
import torch.distributed as dist

class THINGSDatasetCrossAtt(Dataset):
    
    def __init__(
        self,
        d_cfg,
        keep_in_memory = False,
    ):
        self.cfg = d_cfg
        self.name = d_cfg.name
        self.dataset_name = d_cfg.name
        self.trainng = d_cfg.training
        self.use_sampler = True
        self.subjects = d_cfg.subject_num
        self.keep_in_memory = keep_in_memory
        self.collate_fn = collate_fn
        self.worker_init_fn =  None

        # Image transforms producing vision_pixels
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(d_cfg.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(d_cfg.resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )


        if(d_cfg["split"] == "train"):
            self.data = make_train_dataset(self, d_cfg)
        else:
            self.data = make_test_dataset(self, d_cfg)

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index: int):
        row = self.data[index]

        image = row["image"]
        id_ = row["img_id"]
        raw_captions = row["caption"] if self.cfg.good_captions else "image of " + row["text"]

        num_samples = len(raw_captions) if isinstance(raw_captions, list) else 1
        id_txt = [id_] * num_samples

        vision_pixels = self.image_transforms(image.convert("RGB"))

        eeg_np = np.asarray(row["eeg"], dtype=np.float32)  # shape [63, 250]
        conditioning_pixel_values = torch.from_numpy(eeg_np)

        subject = torch.as_tensor(int(row["subject"])) if isinstance(row["subject"], (str, bytes)) else torch.as_tensor(row["subject"])
        label = torch.as_tensor(int(row["label"]))

        # Features (make 1D tensors)
        text_last_hidden_states = torch.as_tensor(row["text_last_hidden_state"])
        text_attention_masks = torch.as_tensor(row["text_attention_mask"])
        img_last_hidden_states = torch.as_tensor(row["img_last_hidden_state"])
        img_pooler_outputs = torch.as_tensor(row["img_pooler_output"])
        

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
            text_attention_masks,
            img_last_hidden_states,
            img_pooler_outputs
        )

   



def collate_fn(examples):

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
            "text_last_hidden_states",
            "text_attention_masks",
            "img_last_hidden_states",
            "img_pooler_outputs"
            ]

    for key, data in zip(keys, all_data):
        if data[0] is None:
            continue 
        elif isinstance(data[0], torch.Tensor):
            batch[key] = torch.stack(data, dim=0)#.float()
        else:
            batch[key] = data
    return batch



def make_train_dataset(
    self,
    d_cfg,
    keep_in_memory = False,
):
    """
    Load train datasets splits
    and then concatenate
    """
    print("Loading THINGS EEG train dataset from Hub...")
    subjects = d_cfg.subject_num
    print("List of subjects: ", subjects)
    parts = []
    for subject in subjects:
        if d_cfg.good_captions:
            dataset = load_dataset(
                "perottofederico/things-eeg-captions-3",
                split="test_sub_08",
                keep_in_memory=keep_in_memory,
            )
        else:
            dataset = load_dataset(
                "perottofederico/things-eeg",
                split="train_"+subject,
                keep_in_memory=keep_in_memory,
            )
        parts.append(dataset)
    dataset = concatenate_datasets(parts)


    def preprocess(split):

        image_transforms = transforms.Compose(
            [
                transforms.Resize(self.cfg.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.cfg.resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        images = [image.convert("RGB") for image in split["image"]]
        images = [image_transforms(image) for image in images]
        split["pixel_values"] = images
        return split

    #if dist.get_rank() == 0:
    #    #dataset = dataset.map(preprocess, batched=True)
    #    dataset = dataset.with_format("torch")
    
    # Optional class filtering
    #if d_cfg.classes_to_remove:
    #    remove_set = set(classes_to_remove)
    #    self.ds = self.ds.filter(lambda x: x["label"] not in remove_set)
    #if d_cfg.classes_to_keep:
    #    keep_set = set(classes_to_keep)
    #    self.ds = self.ds.filter(lambda x: x["label"] in keep_set)
    
    return dataset

    #return HFThingsEEGDataset(
    #    repo_id=repo_id,
    #    split="train",
    #    subjects=subjects,
    #    image_size=image_size,
    #    classes_to_remove=classes_to_remove,
    #    classes_to_keep=None,
    #    keep_in_memory=keep_in_memory,
    #    cache_dir=cache_dir,
    #)


def make_test_dataset(
    self,
    d_cfg,
    keep_in_memory = False,
):
    """
    laod test dataset
    """
    print("Loading THINGS EEG test dataset from Hub...")
    subjects = d_cfg.subject_num
    print("List of subjects: ", subjects)
    parts = []
    for subject in subjects:
        if d_cfg.good_captions:
            dataset = load_dataset(
                "perottofederico/things-eeg-captions-2",
                split="test_sub_08",
                keep_in_memory=keep_in_memory,
            )
        else:
            dataset = load_dataset(
                "perottofederico/things-eeg",
                split="test_"+subject,
                keep_in_memory=keep_in_memory,
            )
        parts.append(dataset)
    dataset = concatenate_datasets(parts)

    # Optional class filtering
    #if classes_to_remove:
    #    remove_set = set(classes_to_remove)
    #    self.ds = self.ds.filter(lambda x: x["label"] not in remove_set)
    #if classes_to_keep:
    #    keep_set = set(classes_to_keep)
    #    self.ds = self.ds.filter(lambda x: x["label"] in keep_set)
    
    return dataset

    #return HFThingsEEGDataset(
    #    repo_id=repo_id,
    #    split="test",
    #    subjects=None,
    #    image_size=image_size,
    #    classes_to_remove=None,
    #    classes_to_keep=None,
    #    keep_in_memory=keep_in_memory,
    #    cache_dir=cache_dir,
    #)





    #  transforms (vision_pixels)
    #self.image_transforms = transforms.Compose(
    #    [
    #        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
    #        transforms.CenterCrop(image_size),
    #        transforms.ToTensor(),
    #        transforms.Normalize([0.5], [0.5]),
    #    ]
    #)
