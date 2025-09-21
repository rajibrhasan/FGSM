import os
import argparse
from ntpath import exists
import torch
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from models import create_model
from datasets import create_dataset
from utils import evaluate, load_best_checkpoint, load_checkpoint, save_checkpoint, set_seed

def train_one_epoch(model, dataloader, criterion, optimizer, device, current_epoch, total_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(dataloader, desc=f"Epoch: {current_epoch}/{total_epochs}"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train(args, model, optimizer, criterion, train_loader, val_loader, device):
    start_epoch = 1
    best_val_acc = 0.0

    checkpoint_path = os.path.join(args.checkpoint_dir, 'model.pth')

    if args.restart and os.path.exists(checkpoint_path):
        start_epoch, best_val_acc = load_checkpoint(model, optimizer, checkpoint_path, device)
    for epoch in range(start_epoch, args.epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
    
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch + 1, best_val_acc, f"{args.checkpoint_dir}/best_model.pth")

        save_checkpoint(model, optimizer, epoch + 1, best_val_acc, checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description="Train a model for clean accuracy")
    parser.add_argument("--dir", type = str, default = 'data', help = "Dataset folder")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--model", type=str, help="Model name from timm")
    parser.add_argument("--checkpoint_dir", type=str, help="Checkpoints directory to load or store checkpoints")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--restart", action="store_true", help="Restart the training if true, else from scratch")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(420)

    # Create model and get transforms
    # model, train_transforms, val_transforms = create_model(args.model, args.num_classes)
    model = create_model(args.model, args.num_classes)
    model.to(device)

    # Load dataset
    train_ds = create_dataset(args.dir, args.dataset)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    val_ds = create_dataset(args.dir, args.dataset, train = False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(args, model, optimizer, criterion, train_loader, val_loader, device)

    
if __name__ == "__main__":
    main()
