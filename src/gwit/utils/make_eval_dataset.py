import os
os.environ['HF_HOME'] = './cache/'
import random
import numpy as np
import torch
import torch.nn.functional as F
import sys
from datasets import load_dataset
from torchvision import transforms
from accelerate.logging import get_logger
import hashlib

logger = get_logger(__name__)


def make_eval_dataset(args, tokenizer, accelerator):
    # Load the tokenizer
    #print("Loading Tokenizer")
    #tokenizer = AutoTokenizer.from_pretrained(
    #        "stabilityai/stable-diffusion-2-1-base",
    #        subfolder="tokenizer",
    #        #revision=args.revision,
    #        use_fast=False,
    #)
    #print("Done!")

    dataset = load_dataset(
            "luigi-s/EEG_Image_CVPR_ALL_subj", 
            split="validation" if "CVPR" in args.dataset_name else "test",
            #args.dataset_name,
            #args.dataset_config_name,
        )#.with_format(type='torch')
        
    # hacky workaround to add an id to the dataset
    # and have the same img_id for the same images
    #TODO consider moving this to the preprocess_train function
    def add_img_id(example):
        arr = np.array(example["image"]) # H×W×C uint8
        example["img_id"] = hashlib.md5(arr.tobytes()).hexdigest()
        return example
    dataset = dataset.map(add_img_id)

    if args.subject_num != 0:
        dataset = dataset.filter(lambda x: x['subject'] == args.subject_num)

    column_names = dataset.column_names

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

    def tokenize_captions(examples, is_train=False):
        # iterate over examples[caption_column] an add half of the captions as empty strings
        # and the other half as the original caption in the captions list
        captions = []
        for caption in examples[caption_column]:
            if random.random() < args.proportion_empty_prompts:
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
      
    def get_good_caption(images):
        from transformers import BlipProcessor, BlipForConditionalGeneration
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")

        captions = []
        for i, image in enumerate(images):
            inputs = processor(images=image, return_tensors="pt").to("cuda")
            out = model.generate(**inputs, max_new_tokens=30, num_beams=4, min_length=5)
            caption = processor.decode(out[0], skip_special_tokens=True)
            captions.append(caption)
        return captions

    def get_caption_from_classifier(eeg, labels):
        # Get the current file path and directory
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)

        # Go up two levels from the current directory
        base_dir = os.path.dirname(os.path.dirname(current_dir))
        path_to_append = base_dir+f"/EEGStyleGAN-ADA/EEG2Feat/Triplet_LSTM/CVPR40" if "CVPR" in args.dataset_name else base_dir+f"/EEGStyleGAN-ADA/EEG2Feat/Triplet_LSTM/Thoughtviz"
        sys.path.append(path_to_append)
        from network import EEGFeatNet
        if "CVPR" in args.dataset_name:
            from dataset_EEG.name_map_ID import id_to_caption
        else:
            from dataset_EEG.name_map_ID import id_to_caption_TVIZ as id_to_caption
        model = EEGFeatNet(n_features=128, projection_dim=128, num_layers=4).to("cuda") if "CVPR" in args.dataset_name else  \
                EEGFeatNet(n_classes=10, in_channels=14, n_features=128, projection_dim=128, num_layers=4).to("cuda")
        model = torch.nn.DataParallel(model).to("cuda")
        import pickle

        # Load the model from the file
        pkl_path = base_dir+'/gwit/dataset_EEG/knn_model.pkl' if "CVPR" in args.dataset_name else base_dir+'/gwit/dataset_EEG/knn_model_TVIZ.pkl'
        with open(pkl_path, 'rb') as f:
            knn_cv = pickle.load(f)
        ckpt_path = base_dir+"/EEGStyleGAN-ADA/EEG2Feat/Triplet_LSTM/CVPR40/EXPERIMENT_29/bestckpt/eegfeat_all_0.9665178571428571.pth" if "CVPR" in args.dataset_name \
            else base_dir+'/EEGStyleGAN-ADA/EEG2Feat/Triplet_LSTM/Thoughtviz/EXPERIMENT_1/bestckpt/eegfeat_all_0.7212357954545454.pth' 
        model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
        
        eeg =  torch.stack(eeg) if "CVPR" in args.dataset_name else torch.stack([torch.tensor(eeg_e) for eeg_e in eeg]) # stack all the eegs
        x_proj = model(eeg.view(-1,eeg.shape[2],eeg.shape[1]).to("cuda")) # reshape the eegs and pass them to the EEGFeatNet model
        labels = [torch.tensor(l) if not isinstance(l, torch.Tensor) else l for l in labels] # convert the labels to tensors (if they aren' already)
        # Predict the labels
        predicted_labels = knn_cv.predict(x_proj[0].cpu().detach().numpy())
        captions = ["image of " + id_to_caption[label] for label in predicted_labels] # add "image of" to the labels
        return captions

    def preprocess_train(train_split):
        image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        # print("\n---------Preprocessing train split\n")
        images = [image.convert("RGB") for image in train_split[image_column]] #Convert each pixel of each image in image_column to 3 8 bit values (via PIL)
        images = [image_transforms(image) for image in images] #Apply the transforms to each image
        #EEG
        conditioning_images = [torch.tensor(image) for image in train_split[conditioning_image_column]] # transform all the conditioning images (eegs) to tensors

        train_split["pixel_values"] = images # Add the pixel values to the train_split 
        train_split["conditioning_pixel_values"] = conditioning_images # Add the conditioning pixel values to the train_split 

        #ids = [id for id in train_split['label_folder']] # get the ids of the images
        #train_split["ids"] = ids # Add the ids to the train_split

        # TO make fixed the captions for EEG
        #if args.caption_fixed:
        #    train_split[caption_column] = len(train_split[caption_column])*[args.caption_fixed_string]
        if args.caption_from_classifier:
            eeg_key = "conditioning_pixel_values" if "CVPR" in args.dataset_name else "eeg_no_resample"
            #train_split[caption_column] = get_caption_from_classifier(train_split[eeg_key], train_split["label"]) # pass to the helper function the eegs (in tensor form) and the labels

        # testing the get_good_caption function
        train_split[caption_column] = get_good_caption(train_split[image_column])
        train_split["input_ids"], train_split["attention_mask"] = tokenize_captions(train_split) # Tokenize the captions we have generated
        return train_split

    # Set the transforms
    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset = dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))
        eval_dataset = dataset.with_transform(preprocess_train)
    #print("Dataset Features: ", eval_dataset.features)
    #print("Images: ", eval_dataset[0]['pixel_values'].shape)
    #print("EEG: ", eval_dataset[0]['conditioning_pixel_values'].shape)
    #print("Text: ", eval_dataset[0]['input_ids'].shape)
    #print("Attention mask: ", eval_dataset[0]['attention_mask'].shape)
    return eval_dataset

def eval_collate_fn(examples): #examples is the batch
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    
    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    attention_mask = torch.stack([example["attention_mask"] for example in examples])
    attention_mask = attention_mask.to(memory_format=torch.contiguous_format).float()

    subjects = torch.stack([torch.as_tensor(example["subject"]) for example in examples])

    #raw_captions = [example["caption"] for example in examples]
    raw_captions = [example["caption"] for example in examples]

    labels = [example["label"] for example in examples] 
    ids = [example["img_id"] for example in examples]
    num_samples = len(raw_captions) if isinstance(raw_captions, list) else 1
    ids_txt = ids.copy()
    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "eeg_subjects": subjects,
        "raw_captions": raw_captions,
        "labels": labels,
        "eeg_subjects": subjects,
        "ids": ids,
        "ids_txt": ids_txt
    }