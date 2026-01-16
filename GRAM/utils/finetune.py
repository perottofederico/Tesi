import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from easydict import EasyDict as edict
from utils.save import ModelSaver
import os
@torch.no_grad()

def finetune_test(model, optimizer, train_loader, val_loaders, args, start_step=0, verbose_time=False):
    for task, val_loader in val_loaders.items():
        print(task)
        tasks = task.split('--')[0]
        print(f"Tasks: {tasks}")
        feat_t = []
        feat_v=[]

        for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            batch = edict(batch)
            evaluation_dict= model(batch, tasks, compute_loss=False)
            feat_t.append(evaluation_dict['feat_t'])
            feat_v.append(evaluation_dict['feat_v'])

        feat_t = torch.cat(feat_t, dim = 0)
        feat_v = torch.cat(feat_v, dim = 0)

        for tensor_t, tensor_v in zip(feat_t, feat_v):
            cosine_TV = torch.matmul(tensor_t, tensor_v.permute(1,0))
            print(f"Cosine similarity between text and video features: {cosine_TV}")
            cosine_VT = torch.matmul(tensor_v, tensor_t.permute(1,0))
            print(f"Cosine similarity between video and text features: {cosine_VT}")

def finetune(model, optimizer, train_loader, val_loaders, args, start_step=0, verbose_time=False):

    best_val_acc = 0.0
    best_val_epoch = 0
    criterion = nn.CrossEntropyLoss()
    model_saver = ModelSaver(os.path.join(args.run_cfg.output_dir, 'classification_finetune'),remove_before_ckpt=args.run_cfg.remove_before_ckpt)
    
    #isolate eeg encoder and add classification head
    new_layer = nn.Sequential(
        nn.Linear(512, 256), #512 = embedding dim
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 40),
        #nn.Softmax(dim=1)
    )
    model = model.eeg_encoder
    model = nn.Sequential(
        model,
        new_layer
    )
    model = model.to(torch.device("cuda"))
    print(model) 

    global_step = start_step
    running_loss = 0.0
    epoch_loss = 0.0
    steps_in_epoch = 0
    '''
    pbar = tqdm(total=args.run_cfg.num_train_steps, initial=start_step)
    for step, (name, batch) in enumerate(train_loader):
        model.train()
        eeg_data = batch['conditioning_pixel_values']
        eeg_data = eeg_data.view(-1,eeg_data.shape[2],eeg_data.shape[1])
        labels = batch['labels']

        inputs, labels = eeg_data.to(torch.device("cuda")), labels.to(torch.device("cuda"))

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_loss += loss.item()
        steps_in_epoch += 1

        pbar.update(1)
        
        if global_step == 0 or (global_step+1) % args.run_cfg.valid_steps == 0:
            #pbar.set_description(f"Step {global_step+1}, Loss: {running_loss / (step + 1):.4f}")
            avg_train_loss = epoch_loss / steps_in_epoch
            print(f"\nEpoch {int((global_step+1) / args.run_cfg.valid_steps)}, Training Loss: {avg_train_loss:.4f}")

            val_acc = evaluate_finetuning(model, val_loaders)
            if best_val_acc < val_acc:
                best_val_acc = val_acc
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

    pbar.close()


    '''
    checkpoint = torch.load(f'{args.run_cfg.output_dir}/best_acc_17352.pth', map_location=torch.device("cuda"))
    model.load_state_dict(checkpoint['model_state_dict'])
    evaluate_finetuning(model, val_loaders)
    

def top_k(predicted, labels, k):
    _, top_k = torch.topk(predicted, k, dim=1)
    correct = top_k.eq(labels.view(-1, 1).expand_as(top_k))
    correct_k = correct.view(-1).float().sum(0, keepdim=True)
    accuracy = 100.0 * correct_k / labels.size(0)
    return accuracy.item()

def evaluate_finetuning(model, val_loader):
    model.eval()
    for task, loader in val_loader.items():
        val_outputs = []
        val_labels = []
        with torch.no_grad():
            for i, batch in tqdm(enumerate(loader), total=len(loader)):
                eeg_data = batch["conditioning_pixel_values"]
                eeg_data = eeg_data.view(-1,eeg_data.shape[2],eeg_data.shape[1])
                labels = batch["labels"]

                eeg_data = eeg_data.to(torch.device("cuda"))
                labels = labels.to(torch.device("cuda"))

                outputs = model(eeg_data)
                val_outputs.append(outputs)
                val_labels.append(labels)

            val_outputs = torch.cat(val_outputs, dim=0)
            val_labels = torch.cat(val_labels, dim=0)

            top1_acc = top_k(val_outputs, val_labels, k=1)
            top5_acc = top_k(val_outputs, val_labels, k=5)
            top10_acc = top_k(val_outputs, val_labels, k=10)
            print(f"Validation Accuracy: Top-1: {top1_acc:.2f}%, Top-5: {top5_acc:.2f}%, Top-10: {top10_acc:.2f}%")    

    return top1_acc