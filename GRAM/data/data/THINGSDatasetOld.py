import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import clip
from torch.nn import functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import requests
import hashlib
from tqdm import tqdm

from transformers import CLIPImageProcessor, CLIPModel
vision_encoder = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
image_processor = CLIPImageProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
vision_dim = 1024 # ViT-H-14 has 1280 hidden size
vision_encoder.requires_grad_(False)  # Freeze the vision encoder

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# vlmodel, preprocess = clip.load("ViT-B/32", device=device)
model_type = 'ViT-H-14'
import open_clip
vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
    model_type, pretrained='laion2b_s32b_b79k', precision='fp32', device=device)

import json

# Load the configuration from the JSON file
#config_path = "data_config.json"
#with open(config_path, "r") as config_file:
#    config = json.load(config_file)

# Access the paths from the config
data_path = "/home/mushy/GRAM_cleaned/GRAM_main/data/things_eeg/Preprocessed_data_250Hz" #config["data_path"]
img_directory_training = "/home/mushy/GRAM_cleaned/GRAM_main/data/things_eeg/training_images" #config["img_directory_training"]
img_directory_test = "/home/mushy/GRAM_cleaned/GRAM_main/data/things_eeg/test_images" #config["img_directory_test"]


class THINGSDatasetOld():
    """
    subjects = ['sub-01', 'sub-02', 'sub-05', 'sub-04', 'sub-03', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']
    """
    def __init__(self, data_path, adap_subject=None, subjects=None, train=True, time_window=[0, 1.0], classes=None, pictures=None):
        self.data_path = data_path #should get passed by config file.
        self.train = train
        self.subject_list = os.listdir(data_path)
        self.subjects = self.subject_list if subjects is None else subjects #should also get passed by config file.
        self.n_sub = len(self.subjects)
        self.time_window = time_window
        self.n_cls = 1654 if train else 200
        self.classes = classes
        self.pictures = pictures
        self.adap_subject = adap_subject  # Save this parameter

        # Assert any subjects in subject_list
        assert any(sub in self.subject_list for sub in self.subjects)

        self.data, self.labels, self.text, self.img, self.subjects = self.load_data()
        
        self.data = self.extract_eeg(self.data, time_window)
        
        # Precompute img_id (sha256) for each unique image in self.img (aligned to self.img list)
        self.img_ids = self._compute_image_ids(self.img)

        if self.classes is None and self.pictures is None:
            # Try to load the saved features if they exist
            print(os.path.join(f'{model_type}'))
            features_filename = os.path.join(f'{model_type}_features_train.pt') if self.train else os.path.join(f'{model_type}_features_test.pt')
            
            if os.path.exists(features_filename):
                print("Loading precomputed features from", features_filename)
                saved_features = torch.load(features_filename)
                self.text_features = saved_features['text_features']
                self.img_features = saved_features['img_features']
            else:
                self.text_features = self.Textencoder(self.text)
                self.img_features = self.ImageEncoder(self.img)
                torch.save({
                    'text_features': self.text_features.cpu(),
                    'img_features': self.img_features.cpu(),
                }, features_filename)
        else:
            print("Computing features...")
            self.text_features = self.Textencoder(self.text)
            self.img_features = self.ImageEncoder(self.img)
            
    def load_data(self):
        print("Loading data...")
        data_list = []
        label_list = []
        texts = []
        images = []
        subjects = []  #subject string per pre-flattened sample
        
        if self.train:
            directory = img_directory_training
        else:
            directory = img_directory_test
        # Get all directories in the path
        dirnames = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        dirnames.sort()
        
        if self.classes is not None:
            dirnames = [dirnames[i] for i in self.classes]

        for dir in dirnames:
            # Try to find the first occurrence of '_'
            try:
                idx = dir.index('_')
                description = dir[idx+1:]  # Get all content after the first '_'
            except ValueError:
                print(f"Skipped: {dir} due to no '_' found.")
                continue
                
            new_description = f"{description}"
            texts.append(new_description)

        if self.train:
            img_directory = img_directory_training  # Replace with your new address
        else:
            img_directory = img_directory_test
        
        all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
        all_folders.sort()  # Ensure the order of folders

        if self.classes is not None and self.pictures is not None:
            images = []  # Initialize images list
            for i in range(len(self.classes)):
                class_idx = self.classes[i]
                pic_idx = self.pictures[i]
                if class_idx < len(all_folders):
                    folder = all_folders[class_idx]
                    folder_path = os.path.join(img_directory, folder)
                    all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    all_images.sort()
                    if pic_idx < len(all_images):
                        images.append(os.path.join(folder_path, all_images[pic_idx]))
        elif self.classes is not None and self.pictures is None:
            images = []  # Initialize images list
            for i in range(len(self.classes)):
                class_idx = self.classes[i]
                if class_idx < len(all_folders):
                    folder = all_folders[class_idx]
                    folder_path = os.path.join(img_directory, folder)
                    all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    all_images.sort()
                    images.extend(os.path.join(folder_path, img) for img in all_images)
        elif self.classes is None:
            images = []  # Initialize images list
            for folder in all_folders:
                folder_path = os.path.join(img_directory, folder)
                all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.sort()  
                images.extend(os.path.join(folder_path, img) for img in all_images)
        else:
            # Handle other cases, such as mismatched lengths of self.classes and self.pictures
            print("Error")
            
        print("self.subjects", self.subjects)
        print("adap_subject", self.adap_subject)
        for subject in self.subjects:
            print("Loading subject:", subject)
            if self.train:
                # if subject == self.adap_subject:  # Skip the excluded subject
                #     continue            
                # print("subject:", subject)    
                file_name = 'preprocessed_eeg_training.npy'

                file_path = os.path.join(self.data_path, subject, file_name)
                data = np.load(file_path, allow_pickle=True)
                
                preprocessed_eeg_data = torch.from_numpy(data['preprocessed_eeg_data']).float().detach()                
                times = torch.from_numpy(data['times']).detach()[50:]
                ch_names = data['ch_names']  # Keep as a Python list or encode appropriately

                n_classes = 1654  # Each class contains 10 images
                samples_per_class = 10  # Each class has ten samples
                
                if self.classes is not None and self.pictures is not None:
                    for c, p in zip(self.classes, self.pictures):
                        start_index = c * 1 + p
                        if start_index < len(preprocessed_eeg_data):  # Ensure index is within range
                            preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+1]  # Select only one sample
                            labels = torch.full((1,), c, dtype=torch.long).detach()  # Add class label
                            data_list.append(preprocessed_eeg_data_class)
                            label_list.append(labels)  # Add labels to the label list
                            subjects.extend([subject.split('-')[1]] * 1)  # align with labels before 4x expansion
                elif self.classes is not None and self.pictures is None:
                    for c in self.classes:
                        start_index = c * samples_per_class
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+samples_per_class]
                        labels = torch.full((samples_per_class,), c, dtype=torch.long).detach()  # Add class label
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)
                        subjects.extend([subject.split('-')[1]] * samples_per_class)

                else:
                    for i in range(n_classes):
                        start_index = i * samples_per_class
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+samples_per_class]
                        labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()  # Add class label
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)
                        subjects.extend([subject.split('-')[1]] * samples_per_class)

                 
            else:
                if subject == self.adap_subject or self.adap_subject == None:  
                    file_name = 'preprocessed_eeg_test.npy'
                    file_path = os.path.join(self.data_path, subject, file_name)
                    data = np.load(file_path, allow_pickle=True)
                    preprocessed_eeg_data = torch.from_numpy(data['preprocessed_eeg_data']).float().detach()
                    times = torch.from_numpy(data['times']).detach()[50:]
                    ch_names = data['ch_names']  # Keep as a Python list or encode appropriately
                    n_classes = 200  # Each class contains 1 image
                    
                    samples_per_class = 1  # Each class has one sample

                    for i in range(n_classes):
                        if self.classes is not None and i not in self.classes:  # Skip if class not in the specified list
                            continue
                        start_index = i * samples_per_class  # Update start_index for each class
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index:start_index+samples_per_class]
                        labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()  # Add class labels
                        preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class.squeeze(0), 0)
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)  # Add labels to the label list
                        subjects.extend([subject.split('-')[1]] * samples_per_class)
                else:
                    continue
        # Data list: (subjects * classes) * (10 * 4 * 17 * 100)
        # Data tensor: (subjects * classes * 10 * 4) * 17 * 100
        if self.train:
            data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[2:])                 
            print("data_tensor", data_tensor.shape)
        else:           
            data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape)   
        label_tensor = torch.cat(label_list, dim=0)
        
        if self.train:
            # Label tensor: (subjects * classes * 10 * 4)
            label_tensor = label_tensor.repeat_interleave(4)
            # expand subjects 4x to match the 4 repetitions
            subjects_per_sample = [s for s in subjects for _ in range(4)]
            if self.classes is not None:
                unique_values = list(label_tensor.numpy())
                lis = []
                for i in unique_values:
                    if i not in lis:
                        lis.append(i)
                    unique_values = torch.tensor(lis)        
                    mapping = {val.item(): index for index, val in enumerate(unique_values)}   
                    label_tensor = torch.tensor([mapping[val.item()] for val in label_tensor], dtype=torch.long)

        else:
            subjects_per_sample = subjects     

        self.times = times
        self.ch_names = ch_names

        print(f"Data tensor shape: {data_tensor.shape}, label tensor shape: {label_tensor.shape}, text length: {len(texts)}, image length: {len(images)}")
        
        return data_tensor, label_tensor, texts, images, subjects_per_sample

    def extract_eeg(self, eeg_data, time_window):
        print("Extracting EEG data in the time window:", time_window)
        start, end = time_window

        # Get the indices of the times within the specified window
        indices = (self.times >= start) & (self.times <= end)
        # Use these indices to select the corresponding data
        extracted_data = eeg_data[..., indices]
        print(f"extracted_data shape: {extracted_data.shape}")

        return extracted_data
    
    def Textencoder(self, text):   
            # Use the preprocessor to convert text to the model's input format
            text_inputs = torch.cat([clip.tokenize(t) for t in text]).to(device)

            # Use the CLIP model to encode text
            with torch.no_grad():
                text_features = vlmodel.encode_text(text_inputs)
            
            text_features = F.normalize(text_features, dim=-1).detach()
       
            return text_features
        
    def ImageEncoder(self, images):
        batch_size = 20  # Set to an appropriate value
        image_features_list = []
      
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            image_inputs = torch.stack([preprocess_train(Image.open(img).convert("RGB")) for img in batch_images]).to(device)

            with torch.no_grad():
                batch_image_features = vlmodel.encode_image(image_inputs)
                batch_image_features /= batch_image_features.norm(dim=-1, keepdim=True)

            image_features_list.append(batch_image_features)

        image_features = torch.cat(image_features_list, dim=0)
        
        return image_features
    
    def compute_image_ids(self, image_paths, algo="sha256"):
        """
        Compute a stable hash for each image path in image_paths.
        Returns a list aligned with image_paths.
        """
        def file_hash(path, algo="sha256", buf_size=1024 * 1024):
            h = hashlib.new(algo)
            with open(path, "rb") as f:
                while True:
                    b = f.read(buf_size)
                    if not b:
                        break
                    h.update(b)
            return h.hexdigest()[:16] #16 should be enough hopefully

        ids = []
        cache = {}
        for p in tqdm(image_paths, desc="Hashing images"):
            if p in cache:
                ids.append(cache[p])
            else:
                try:
                    hid = file_hash(p, algo=algo)
                except Exception:
                    hid = None
                cache[p] = hid
                ids.append(hid)
        return ids

    def __getitem__(self, index):
        # Get the data and label corresponding to "index"
        # index: (subjects * classes * 10 * 4)
        x = self.data[index]
        label = self.labels[index]
        
        if self.pictures is None:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 10 * 4
                index_n_sub_test = self.n_cls * 1 * 80
            else:
                index_n_sub_test = len(self.classes)* 1 * 80
                index_n_sub_train = len(self.classes)* 10 * 4
            # text_index: classes
            if self.train:
                text_index = (index % index_n_sub_train) // (10 * 4)
            else:
                text_index = (index % index_n_sub_test)
            # img_index: classes * 10
            if self.train:
                img_index = (index % index_n_sub_train) // (4)
            else:
                img_index = (index % index_n_sub_test)
        else:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 1 * 4
                index_n_sub_test = self.n_cls * 1 * 80
            else:
                index_n_sub_test = len(self.classes)* 1 * 80
                index_n_sub_train = len(self.classes)* 1 * 4
            # text_index: classes
            if self.train:
                text_index = (index % index_n_sub_train) // (1 * 4)
            else:
                text_index = (index % index_n_sub_test)
            # img_index: classes * 10
            if self.train:
                img_index = (index % index_n_sub_train) // (4)
            else:
                img_index = (index % index_n_sub_test)
                
        text = self.text[text_index]
        img = self.img[img_index]
        
        text_features = self.text_features[text_index]
        img_features = self.img_features[img_index]
        
        subject = self.subjects[index]
        img_id = self.img_ids[img_index] if 0 <= img_index < len(self.img_ids) else None

        return x, label, text, text_features, img, img_features, subject, img_id

    def __len__(self):
        return self.data.shape[0]  # or self.labels.shape[0] which should be the same

if __name__ == "__main__":
    # Instantiate the dataset and dataloader
    # data_path = "/home/ldy/Workspace/THINGS/EEG/osfstorage-archive"  # Replace with the path to your data
    data_path = data_path
    #train_dataset = THINGSDataset(data_path, subjects=['sub-01'], train=True)    
    test_dataset = THINGSDatasetOld(data_path, subjects=None, train=False)
    # train_dataset = THINGSDataset(data_path, adap_subject='sub-01', train=True)    
    # test_dataset = THINGSDataset(data_path, adap_subject='sub-01', train=False)    
    # train_dataset = THINGSDataset(data_path, train=True) 
    # test_dataset = THINGSDataset(data_path, train=False) 
    # Training EEG data shape: torch.Size([16540, 4, 17, 100]) [Number of training images, repetition count, channels, EEG time points]
    # Testing EEG data shape: torch.Size([200, 80, 17, 100])
    # 1 second 'times': array([-0.2 , -0.19, -0.18, ... , 0.76,  0.77,  0.78, 0.79])}
    # 17 channels 'ch_names': ['Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2']
    # 100 Hz
    #train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    i = 80*1-1
    x, label, text, text_features, img, img_features, subject, img_id = test_dataset[i]
    print(f"Index {i}, Label: {label}, text: {text}, eeg shape: {x.shape}")
    print("subject:", subject, "img_id:", img_id, "text:", text, "shape:", x.shape, "img_path:", img)
    #im = Image.open(img)
    #im.save("sample_image_from_testset.jpg")