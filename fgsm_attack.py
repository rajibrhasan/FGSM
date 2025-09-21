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
from utils import load_best_checkpoint, evaluate, set_seed

def get_random_targets(true_labels, num_classes):
    targets = []
    for t in true_labels.cpu().tolist():
        choices = set(range(num_classes)) - set([t])
        targets.append(random.choice(list(choices)))
    
    return torch.tensor(targets, dtype = true_labels.dtype, device = true_labels.device)


def targeted_fgsm_attack(data_loader, model, criterion, num_classes, eps, device, target_type):
    model.eval()
    robust_correct = 0
    total = 0

    for images, labels in tqdm(data_loader, f'targeted_{target_type}(ε={eps: .4f}): '):
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True

        outputs = model(images)
        _, clean_preds = torch.max(outputs, 1)

        if target_type == 'least_likely':
            target_labels = outputs.argmin(dim = 1)
        elif target_type == 'random':
            target_labels = get_random_targets(labels, num_classes)

        mask = (clean_preds != target_labels)
        if not mask.any():
            continue
            
        loss = criterion(outputs, target_labels)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            adv_images = torch.clamp(images - eps * images.grad.sign(), 0, 1)
            robust_outputs = model(adv_images)
            _, robust_preds = torch.max(robust_outputs, 1)
            robust_correct += ((robust_preds == target_labels) & mask).sum().item()
            total += mask.sum().item()
    if total == 0:
        return 0.0, 0.0
    
    asr = robust_correct / total

    return asr



def fgsm_attack(data_loader, model, criterion, eps, device):
    model.eval()
    robust_correct = 0
    total = 0

    for images, labels in tqdm(data_loader, f'Untargeted(ε={eps: 0.4f}): '):
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True

        outputs = model(images)
        _, clean_preds = torch.max(outputs, 1)

        loss = criterion(outputs, labels)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            adv_images = torch.clamp(images + eps * images.grad.sign(), 0, 1)
            robust_outputs = model(adv_images)
            _, robust_preds = torch.max(robust_outputs, 1)
            robust_correct += (robust_preds == labels).sum().item()
            total += labels.size(0)

    robust_acc = robust_correct / total
    asr = 1 - robust_acc

    return robust_acc, asr



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
    load_best_checkpoint(model, checkpoint_path, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    results = []
   
    eps_list = [ 1/255, 2/255, 4/255, 8/255]

    for eps in eps_list:
        robust_acc, asr = fgsm_attack(val_loader, model, criterion, eps, device)
        t_asr_r = targeted_fgsm_attack(val_loader, model, criterion, args.num_classes, eps, device, "random")
        t_asr_l = targeted_fgsm_attack(val_loader, model, criterion, args.num_classes, eps, device, "least_likely")
        results.append([f"ε={eps:.4f}", f"{val_acc * 100: 0.2f}", f"{robust_acc*100:.2f}", f"{asr*100: .2f}", f"{t_asr_r*100: .2f}", f"{t_asr_l*100:.2f}"])
    
    headers = ["Setting", "Clean Acc", "Robust Acc", "Untargeted ASR", "Targeted_ASR(random)", "Targeted_ASR(least_likely)"]
    # table = tabulate(results, headers=headers, floatfmt=".2f")
    results_df = pd.DataFrame(results, columns = headers)
    print(results_df)

    # write to file
    with open(f"{args.log_dir}/{args.model}_{args.dataset}.log", "w") as f:
        f.write(tabulate(results_df, headers='keys', tablefmt='grid', floatfmt=".2f"))

    

if __name__== "__main__":
    main()

   





        




