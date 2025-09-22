import torch
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from constant import STATS, CLASS_NAMES
plt.rcParams.update({'font.size': 10})

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc = "Eval: ") :
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def save_checkpoint(model, optimizer, epoch, best_acc, path):
    """Save the current model and optimizer state."""
    state = {
        'epoch': epoch,
        'best_val_acc': best_acc,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(state, path)


def load_checkpoint(model, optimizer=None, path="checkpoints/best_model.pth", device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_val_acc = checkpoint.get('best_val_acc', 0.0)
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint '{path}' (epoch {start_epoch-1})| best_val_acc: {best_val_acc}")
    return start_epoch, best_val_acc

def load_best_checkpoint(model, path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    best_val_acc = checkpoint.get('best_val_acc', 0.0)
    
    print(f"Loaded best checkpoint from {path} | Acc: {best_val_acc}")
    return best_val_acc

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if using CUDA:
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # make CuDNN deterministic (may slow things down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def denormalize(img, mean, std):
    """Denormalize a tensor image using dataset mean & std."""
    mean = torch.tensor(mean).view(3,1,1).to(img.device)
    std = torch.tensor(std).view(3,1,1).to(img.device)
    img = img * std + mean
    return img

def prepare_for_plot(img, mean, std):
    """
    Prepare image for matplotlib.
    """
    # Denormalize
    img = denormalize(img, mean, std)  
    img = img.clamp(0, 1)
    # Convert to numpy (H,W,C)
    img = img.permute(1, 2, 0).cpu().numpy()
    return img

def visualize_util(vis_samples, dset_name, out_dir, target_type, scale = 10):
    
    mean, std = STATS[dset_name]
    class_names = CLASS_NAMES[dset_name]
    attack_name = 'untargeted' if target_type == 'none' else f'targeted_{target_type}'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for key, sample in vis_samples.items():
        if sample is None:
            print(f'No sample for {key} for {dset_name}')
            continue
        orig_img, adv_img = sample['orig_img'], sample['adv_img']
        orig_out, adv_out = sample['orig_out'], sample['adv_out']

        orig_img = prepare_for_plot(orig_img, mean, std)
        adv_img = prepare_for_plot(adv_img, mean, std)


        pert_img = (adv_img - orig_img) 
        pmin, pmax = pert_img.min(), pert_img.max()
        pert_img = (pert_img - pmin) / (pmax - pmin + 1e-8)

        softmax = torch.nn.Softmax(dim=0)
        orig_probs = softmax(orig_out)
        adv_probs  = softmax(adv_out)

        orig_class = torch.argmax(orig_probs).item()
        adv_class  = torch.argmax(adv_probs).item()

        fig, axes = plt.subplots(1,3)

        orig_title = (
            f"Original Image\n"
            f"True Label: {class_names[sample['true_label']]}\n"
            f"Prediction: {class_names[orig_class]}\n"
            f"Confidence: {orig_probs[orig_class]*100:0.2f}%"
        )

        adv_title = (
            f"Adversarial Image\n"
            f"Target Label: {class_names[sample['target_label']]}\n"
            f"Prediction: {class_names[adv_class]}\n"
            f"Confidence: {adv_probs[adv_class]*100:0.2f}%"
        )

        if target_type == "none":
            adv_title = adv_title.replace(f"Target Label: {class_names[sample['target_label']]}\n", "")

       

        axes[0].imshow(orig_img)
        axes[0].axis("off")
        axes[0].set_title(orig_title, fontsize = 8)
        

        axes[1].imshow(pert_img)
        axes[1].axis("off")
        axes[1].set_title(f"Perturbation(X{scale})", fontsize = 8)

        axes[2].imshow(adv_img)
        axes[2].axis("off")
        axes[2].set_title(adv_title, fontsize = 8)

        plt.tight_layout()
        plt.savefig(f"{out_dir}/{attack_name}_{key}.png", dpi=200, bbox_inches="tight")

