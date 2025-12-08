import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.resnet18 import ResNet18
from utils.device import get_device
from utils.dataset import CustomDataset

# ====================
# 1. Configuration
# ====================
data_root = "data"
num_classes = 4
batch_size = 32
num_workers = 4
num_epochs = 20
val_split_ratio = 0.2

device = get_device()

save_dir = Path("checkpoints")
save_dir.mkdir(parents=True, exist_ok=True)


# ====================
# 2. Transforms, Datasets
# ====================
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

full_dataset = CustomDataset(
    data_dir=data_root,
    data_type="train",
    is_augment=False,
)

total_len = len(full_dataset)
val_len = int(total_len * val_split_ratio)
train_len = total_len - val_len

indices = torch.randperm(total_len)
train_indices = indices[:train_len]
val_indices = indices[train_len:]

train_dataset_full = CustomDataset(
    data_dir=data_root,
    data_type="train",
    is_augment=False,
    transform=train_transform,
)

val_dataset_full = CustomDataset(
    data_dir=data_root,
    data_type="train",
    is_augment=False,
    transform=val_transform,
)

train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
val_dataset   = torch.utils.data.Subset(val_dataset_full, val_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
)


# ====================
# 3. Optimizer, Scheduler
# ====================
def create_optimizer(model, exp_cfg):
    opt_type = exp_cfg.get("optimizer", "adam").lower()
    lr = exp_cfg["lr"]
    wd = exp_cfg.get("weight_decay", 0.0)

    if opt_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == "sgd":
        momentum = exp_cfg.get("momentum", 0.9)
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=wd,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")

    return optimizer


def create_scheduler(optimizer, exp_cfg):
    sch_cfg = exp_cfg.get("scheduler", None)
    if sch_cfg is None:
        return None

    sch_type = sch_cfg.get("type", None)
    if sch_type is None:
        return None

    sch_type = sch_type.lower()

    if sch_type == "step":
        step_size = sch_cfg.get("step_size", 7)
        gamma = sch_cfg.get("gamma", 0.1)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    elif sch_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {sch_type}")

    return scheduler


# ====================
# 4. Train, Validate Loops
# ====================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix({
            "loss": f"{running_loss/total:.4f}",
            "acc": f"{correct/total:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
        })

    train_loss = running_loss / total
    train_acc = correct / total

    return train_loss, train_acc


def validation(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc="Validation", leave=False)

    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix({
                "loss": f"{running_loss/total:.4f}",
                "acc": f"{correct/total:.4f}"
            })

    validation_loss = running_loss / total
    validation_acc = correct / total

    return validation_loss, validation_acc


# ====================
# 5. Plot loss/accuracy curves
# ====================
def plot_curves(train_losses, val_losses, train_accs, val_accs, exp_name):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title(f'Loss Curves - ({exp_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'loss_curve_{exp_name}.png')
    plt.close()


    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title(f'Accuracy Curves - ({exp_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'accuracy_curve_{exp_name}.png')
    plt.close()


# ====================
# 6. Main training loop
# ====================
def main():
    cfg_path = Path("config") / "res_cfg.json"
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    experiments = cfg.get("experiments", {})

    summary = []

    criterion = nn.CrossEntropyLoss()

    for exp_cfg in experiments:
        exp_name = exp_cfg.get("name", "exp")
        print(f"\nExperiment: {exp_name}")

        model = ResNet18(num_classes=num_classes, pretrained=False)
        model = model.to(device)

        optimizer = create_optimizer(model, exp_cfg)
        scheduler = create_scheduler(optimizer, exp_cfg)

        best_model_path = save_dir / f"best_{exp_name}.pth"

        best_val_acc = 0.0
        best_val_loss = float("inf")
        best_epoch = -1

        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []

        for epoch in range(num_epochs):
            print(f"\nEpoch: {epoch+1}/{num_epochs}")

            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            print(f"Train | loss: {train_loss:.4f}, acc: {train_acc*100:.2f}%")

            val_loss, val_acc = validation(
                model, val_loader, criterion, device
            )
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            print(f"Val   | loss: {val_loss:.4f}, acc: {val_acc*100:.2f}%")

            if scheduler is not None:
                scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_epoch = epoch + 1
                torch.save({
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": best_val_acc,
                    "val_loss": best_val_loss,
                    "exp_config": exp_cfg,
                }, best_model_path)

                print(f"Best model updated | val loss: {best_val_loss:.4f}, val acc: {best_val_acc*100:.2f}%")

        plot_curves(train_losses, val_losses, train_accs, val_accs, exp_name)

        print(f"\n[{exp_name}] Training completed")
        print(f"\n Best model | epoch: {best_epoch}, val loss: {best_val_loss:.4f}, val acc: {best_val_acc*100:.2f}%")

        summary.append({
            "experiment": exp_name,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
            "best_model_path": str(best_model_path),
        })

    print("\n===== Summary =====")
    for exp_summary in summary:
        print(f"Experiment: {exp_summary['experiment']}")
        print(f"Best Epoch: {exp_summary['best_epoch']}")
        print(f"Best Val Acc: {exp_summary['best_val_acc']*100:.2f}%")
        print(f"Best Val Loss: {exp_summary['best_val_loss']:.4f}")

if __name__ == "__main__":
    main()