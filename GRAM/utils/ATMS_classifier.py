import torch.nn as nn
import torch
from torch.utils.data import random_split, DataLoader
from utils.save import ModelSaver
from tqdm import tqdm
import numpy as np
from easydict import EasyDict as edict
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EEGClassifier(nn.Module):
    def __init__(self,atms_model, num_classes):
        super().__init__()
        self.encoder = atms_model  # load your pretrained weights afterwards
        self.classifier = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Dropout(0.1),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x, subject_ids):
        with torch.no_grad():  # freeze the encoder during training
            features = self.encoder(x, subject_ids)   # [B, 1024]
        logits = self.classifier(features)            # [B, num_classes]
        return logits


def split_train_val(train_loader, val_ratio=0.1, seed=42):
    dataset = train_loader.dataset
    n_total = len(dataset)
    n_val = int(val_ratio * n_total)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    batch_size = train_loader.batch_size
    num_workers = train_loader.num_workers

    train_loader_new = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=getattr(train_loader, "pin_memory", False))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=getattr(train_loader, "pin_memory", False))
    print("Len train_loader, val_loader: ", len(train_loader_new), len(val_loader))
    return train_loader_new, val_loader


def ATMS_classifier(gram_model, optimizer, train_loader, val_loader, args):
    model = EEGClassifier(gram_model.eeg_encoder, num_classes=1654).to(device)
    print("classification head added to ATMS model")
    criterion = nn.CrossEntropyLoss()
    model_saver = ModelSaver(os.path.join(args.run_cfg.output_dir, 'ATMS_classification'),remove_before_ckpt=args.run_cfg.remove_before_ckpt)
    print(gram_model.eeg_encoder)
    #train_loader, val_loader = split_train_val(train_loader, val_ratio=0.1, seed=42)
    print("Training classifier...")
    global_step = 0
    running_loss = 0.0
    epoch_loss =0.0
    steps_in_epoch = 0
    best_val_acc = 0.0
    for step, (name, batch) in enumerate(train_loader):
        model.train()
        eeg_data = batch['conditioning_pixel_values']
        labels   = batch['labels']
        subject_ids = batch['eeg_subjects']
        eeg_data, subject_ids, labels = eeg_data.to(device), subject_ids.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(eeg_data, subject_ids)              # [B, K]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        epoch_loss += loss.item()
        steps_in_epoch += 1
        
        if global_step == 0 or (global_step+1) % args.run_cfg.valid_steps == 0:
            #avg_train_loss = running_loss / len(train_loader)
            #print(f"train loss {avg_train_loss:.4f}")
            #pbar.set_description(f"Step {global_step+1}, Loss: {running_loss / (step + 1):.4f}")
            avg_train_loss = epoch_loss / steps_in_epoch
            print(f"\nEpoch {int((global_step+1) / args.run_cfg.valid_steps)}, Training Loss: {avg_train_loss:.4f}")
            
            top1, top5 = evaluate_classifier(model, val_loader, device)
            print(f"val top‑1 {top1:.2f}%, top‑5 {top5:.2f}%")
            if best_val_acc < top1:
                best_val_acc = top1
                best_val_epoch = step/args.run_cfg.valid_steps
                # Save the best model checkpoint
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': step/args.run_cfg.valid_steps,
                },  f'{args.run_cfg.output_dir}/best_acc_{global_step}.pth')

            epoch_loss = 0.0
            steps_in_epoch = 0
        global_step += 1
        print(f"{global_step}", end='\r')
        
    top1, top5 = evaluate_classifier(model, val_loader, device)
        
    return model


def top_k(predicted, labels, k):
    _, top_k_idx = torch.topk(predicted, k, dim=1)
    correct = top_k_idx.eq(labels.view(-1, 1).expand_as(top_k_idx))
    correct_k = correct.reshape(-1).float().sum(0, keepdim=True)
    return (100.0 * correct_k / labels.size(0)).item()

@torch.no_grad()
def evaluate_classifier(model, val_loader, device):
    model.eval()
    all_logits = []
    all_labels = []
    for task, loader in val_loader.items():
        val_outputs = []
        val_labels = []
        with torch.no_grad():
            for i, batch in tqdm(enumerate(loader), total=len(loader)):
                eeg_data = batch['conditioning_pixel_values']
                labels   = batch['labels']
                subject_ids = batch['eeg_subjects']

                eeg_data   = eeg_data.to(device)
                labels     = labels.to(device)
                subject_ids = subject_ids.to(device)

                logits = model(eeg_data, subject_ids)
                all_logits.append(logits)
                all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    top1 = top_k(all_logits, all_labels, k=1)
    top5 = top_k(all_logits, all_labels, k=5 if 27 >= 5 else 27)
    return top1, top5