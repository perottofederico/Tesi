import os
os.environ['HF_HOME'] = './cache/'
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from datasets import load_dataset
from torchvision import transforms
from accelerate.logging import get_logger
logger = get_logger(__name__)

def make_train_dataset(args, tokenizer, accelerator):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir #"/leonardo_scratch/fast/IscrC_GenOpt/luigi/" #args.cache_dir,
        )
    else:
        if args.train_data_dir is not None:
            dataset = load_dataset(
                args.train_data_dir,
                cache_dir=args.cache_dir,
            )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if args.subject_num != 0:
        dataset = dataset.filter(lambda x: x['subject'] == args.subject_num)

    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.caption_column is None:
        caption_column = column_names[1]
        logger.info(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.conditioning_image_column is None:
        conditioning_image_column = column_names[2]
        logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
    else:
        conditioning_image_column = args.conditioning_image_column
        if conditioning_image_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_column` value '{args.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    def tokenize_captions(examples, is_train=True):
        # iterate over examples[caption_column] an add half of the captions as empty strings
        # and the other half as the original caption in the captions list
        captions = []
        for caption in examples[caption_column]:
            if random.random() < args.proportion_empty_prompts:
                print("Empty prompt")
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        # Tokenize the captions
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        # Return the tokenized captions (shape [1,77])
        return inputs.input_ids, inputs.attention_mask

    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # conditioning_image_transforms = transforms.Compose(
    #     [
    #         transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
    #         transforms.CenterCrop(args.resolution),
    #         transforms.ToTensor(),
    #     ]
    # )
    import sys
    # Get the current file path and directory
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    # Go up two levels from the current directory
    base_dir = os.path.dirname(os.path.dirname(current_dir))
    print(base_dir, "aaaaaaa")
    # print(base_dir+"/EEGStyleGAN-ADA/EEG2Feat/Triplet_LSTM/CVPR40")
    path_to_append = base_dir+f"\\EEGStyleGAN-ADA\\EEG2Feat\\Triplet_LSTM\\CVPR40" if "CVPR" in args.dataset_name else base_dir+f"\\EEGStyleGAN-ADA\\EEG2Feat\\Triplet_LSTM\\Thoughtviz"
    sys.path.append(path_to_append)
    from network import EEGFeatNet
    # sys.path.append(base_dir+"/diffusers/src/dataset_EEG/")
    if "CVPR" in args.dataset_name:
        from dataset_EEG.name_map_ID import id_to_caption
        print("Ids to caption: ", id_to_caption)
    else:
        from dataset_EEG.name_map_ID import id_to_caption_TVIZ as id_to_caption
        print(id_to_caption)
    model = EEGFeatNet(n_features=128, projection_dim=128, num_layers=4).to("cuda") if "CVPR" in args.dataset_name else  \
            EEGFeatNet(n_classes=10, in_channels=14, n_features=128, projection_dim=128, num_layers=4).to("cuda")
    model = torch.nn.DataParallel(model).to("cuda")
    import pickle

    # Load the model from the file
    pkl_path = base_dir+'\\gwit\\dataset_EEG\\knn_model.pkl' if "CVPR" in args.dataset_name else base_dir+'\\gwit\\dataset_EEG\\knn_model_TVIZ.pkl'
    with open(pkl_path, 'rb') as f:
        knn_cv = pickle.load(f)
    ckpt_path = base_dir+"\\EEGStyleGAN-ADA\\EEG2Feat\\Triplet_LSTM\\CVPR40\\EXPERIMENT_29\\bestckpt\\eegfeat_all_0.9665178571428571.pth" if "CVPR" in args.dataset_name \
        else base_dir+'\\EEGStyleGAN-ADA\\EEG2Feat\\Triplet_LSTM\\Thoughtviz\\EXPERIMENT_1\\bestckpt\\eegfeat_all_0.7212357954545454.pth' 
    model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])

    def get_caption_from_classifier(eeg, labels):
        #TODO
        # import pdb
        # pdb.set_trace()
        eeg =  torch.stack(eeg) if "CVPR" in args.dataset_name else torch.stack([torch.tensor(eeg_e) for eeg_e in eeg]) # stack all the eegs
        x_proj = model(eeg.view(-1,eeg.shape[2],eeg.shape[1]).to("cuda")) # reshape the eegs and pass them to the EEGFeatNet model
        labels = [torch.tensor(l) if not isinstance(l, torch.Tensor) else l for l in labels] # convert the labels to tensors (if they aren' already)
        # Predict the labels
        predicted_labels = knn_cv.predict(x_proj.cpu().detach().numpy())
        captions = ["image of " + id_to_caption[label] for label in predicted_labels] # add "image of" to the labels
        return captions

    def preprocess_train(train_split):
        # print(train_split[image_column][0])
        images = [image.convert("RGB") for image in train_split[image_column]] #Convert each pixel of each image in image_column to 3 8 bit values (via PIL)
        images = [image_transforms(image) for image in images] #Apply the transforms to each image

        #TEST
        # conditioning_images = [image.convert("RGB") for image in train_split[conditioning_image_column]]
        # conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]
        #EEG
        conditioning_images = [torch.tensor(image) for image in train_split[conditioning_image_column]] # transform all the conditioning images (eegs) to tensors

        train_split["pixel_values"] = images # Add the pixel values to the train_split 
        train_split["conditioning_pixel_values"] = conditioning_images # Add the conditioning pixel values to the train_split 
        # TO make fixed the captions for EEG
        if args.caption_fixed:
            train_split[caption_column] = len(train_split[caption_column])*[args.caption_fixed_string]
        if args.caption_from_classifier:
            eeg_key = "conditioning_pixel_values" if "CVPR" in args.dataset_name else "eeg_no_resample"
            train_split[caption_column] = get_caption_from_classifier(train_split[eeg_key], train_split["label"]) # pass to the helper function the eegs (in tensor form) and the labels

        train_split["input_ids"], train_split["attention_mask"] = tokenize_captions(train_split) # Tokenize the captions we have generated
        return train_split

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)
    print("Dataset Features: ", train_dataset.features)
    print("Images: ", train_dataset[0]['pixel_values'].shape)
    print("EEG: ", train_dataset[0]['conditioning_pixel_values'].shape)
    print("Text: ", train_dataset[0]['input_ids'].shape)
    print("Attention mask: ", train_dataset[0]['attention_mask'].shape)
    return train_dataset