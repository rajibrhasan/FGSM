import os
from tqdm import tqdm
import torch
import torch.nn as nn
import argparse
from tabulate import tabulate
import random
from datasets import create_dataset
from models import create_model
from torch.utils.data import DataLoader
import pandas as pd
from utils import STATS, load_best_checkpoint, evaluate, set_seed, visualize_util
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def get_random_targets(true_labels, num_classes):
    targets = []
    for t in true_labels.cpu().tolist():
        choices = set(range(num_classes)) - set([t])
        targets.append(random.choice(list(choices)))
    
    return torch.tensor(targets, dtype = true_labels.dtype, device = true_labels.device)


def fgsm_attack(data_loader, model, dset_name, criterion, eps, num_classes, device, target_type='none'):
    model.eval()
    robust_correct = 0
    total = 0
    MEAN, STD = STATS[dset_name]
   

    vis_samples = {
        'success': None,
        'failure': None
    }

    attack_name = 'untargeted' if target_type == 'none' else f'targeted_{target_type}'

    for images, labels in tqdm(data_loader, f'{attack_name}(ε = {eps: 0.4f}): '):
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True

        outputs = model(images)
        _, clean_preds = torch.max(outputs, 1)

        if target_type == 'least_likely':
            target_labels = outputs.argmin(dim = 1)
        elif target_type == 'random':
            target_labels = get_random_targets(labels, num_classes)
        else:
            target_labels = labels
        
        mask = (clean_preds != target_labels) if target_type != 'none' else torch.ones_like(labels, dtype=torch.bool)
        loss = criterion(outputs, target_labels)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            if target_type == 'none':
                adv_images = images + eps * images.grad.sign()
            else:
                adv_images = images - eps * images.grad.sign()
            
            mean = torch.tensor(MEAN, device=device).view(1,3,1,1)
            std  = torch.tensor(STD, device=device).view(1,3,1,1)
            pixel_min = (0.0 - mean) / std
            pixel_max = (1.0 - mean) / std
            adv_images = torch.max(torch.min(adv_images, pixel_max), pixel_min)
            robust_outputs = model(adv_images)
            _, robust_preds = torch.max(robust_outputs, 1)
            robust_correct += ((robust_preds == target_labels) & mask).sum().item()
            total += mask.sum().item()

        keys = ['success', 'failure']

        for key in keys:
            if vis_samples[key] is None:
                if key == 'success':
                    idxs = (robust_preds == target_labels).nonzero(as_tuple=True)[0]
                elif key == 'failure':
                    idxs = (robust_preds != target_labels).nonzero(as_tuple=True)[0]
            

                if len(idxs) > 0:
                    # Randomly pick one failed sample
                    idx = random.choice(idxs).item()
                    vis_samples[key] = {
                        'orig_img': images[idx].detach().cpu(),
                        'adv_img': adv_images[idx].detach().cpu(),
                        'true_label': labels[idx].detach().cpu(),
                        'target_label': target_labels[idx].detach().cpu(),
                        'orig_out': outputs[idx].detach().cpu(),
                        'adv_out': robust_outputs[idx].detach().cpu()
                    }
                
        
        if target_type == 'none':
            temp = vis_samples['success']
            vis_samples['success'] = vis_samples['failure']
            vis_samples['failure'] = temp

       
    
    robust_acc = robust_correct / total
    asr = 1 - robust_acc

    return robust_acc, asr, vis_samples



def main():
    parser = argparse.ArgumentParser(description="FGSM attack on a trained model")
    parser.add_argument("--dir", type = str, default = 'data', help = "Dataset folder")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--model", type=str, help="Model name from timm")
    parser.add_argument("--checkpoint_dir", type=str, help="Checkpoints directory to load the bestcheckpoints")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--log_dir", type=str, default="logs", help="Logging path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(420)


    model = create_model(args.model, args.num_classes)
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False

    val_ds = create_dataset(args.dir, args.dataset, train = False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    criterion = nn.CrossEntropyLoss()

    checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
    val_acc = load_best_checkpoint(model, checkpoint_path, device)
    # val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    # print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    results = []
   
    eps_list = [ 1/255, 2/255, 4/255, 8/255]
    target_types = ['none', 'random', 'least_likely']

    for eps in eps_list:
        row = [f"ε={eps:.4f}", f"{val_acc * 100: 0.2f}"]

        for typ in target_types:
            acc, asr, vis_samples = fgsm_attack(val_loader, model, args.dataset, criterion, eps, args.num_classes, device, typ)
            if typ == 'none':
                row.extend([f"{acc*100:.2f}", f"{asr*100: .2f}"])
            else:
                row.append(f"{acc*100:.2f}")
            
            if eps == 8/255:
                visualize_util(vis_samples, args.dataset,f"asests/{args.dataset}/{args.model}", typ)
        results.append(row)

    
    headers = ["Setting", "Clean Acc", "Robust Acc", "Untargeted ASR", "Targeted_ASR(random)", "Targeted_ASR(least_likely)"]
    # table = tabulate(results, headers=headers, floatfmt=".2f")
    results_df = pd.DataFrame(results, columns = headers)
    print(results_df)
    
    fname = f"{args.log_dir}/{args.model}_{args.dataset}"

    results_df.to_csv(fname+".csv", index = False)

    # write to file
    with open(fname+".log", "w") as f:
        f.write(tabulate(results_df, headers='keys', tablefmt='grid', floatfmt=".2f"))

    

if __name__== "__main__":
    main()

   





        




