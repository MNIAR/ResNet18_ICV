import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.resnet18 import ResNet18
from utils.device import get_device
from utils.dataset import CustomDataset

# ====================
# 1. Configuration
# ====================
data_root = "data"
num_classes = 4
batch_size = 32
num_workers = 0

device = get_device()

save_dir = Path("checkpoints")
save_dir.mkdir(parents=True, exist_ok=True)


# ====================
# 2. Transforms, Datasets
# ====================
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_dataset = CustomDataset(
    data_dir=data_root,
    data_type="test",
    is_augment=False,
    transform=test_transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
)


# ====================
# 3. Evaluation Loop
# ====================
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc="Testing", leave=False)

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
                "acc": f"{correct/total:.4f}",
            })

    test_loss = running_loss / total
    test_acc = correct / total

    return test_loss, test_acc


# ====================
# 4. Test
# ====================
def main():
    cfg_path = Path("config") / "res_cfg.json"
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    experiments = cfg.get("experiments", [])

    summary = []

    criterion = nn.CrossEntropyLoss()

    for exp_cfg in experiments:
        exp_name = exp_cfg.get("name", "exp")
        print(f"\nExperiment (test): {exp_name}")

        best_model_path = save_dir / f"best_{exp_name}.pth"
        model = ResNet18(num_classes=num_classes, pretrained=False)
        model = model.to(device)

        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        print(f"Best epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"Best val acc: {checkpoint.get('val_acc', 0.0)*100:.2f}%")
        print(f"Best val loss: {checkpoint.get('val_loss', 0.0):.4f}")

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"Test | loss: {test_loss:.4f}, acc: {test_acc*100:.2f}%\n")

        summary.append({
            "experiment": exp_name,
            "best_model_path": str(best_model_path),
            "epoch": checkpoint.get("epoch", None),
            "val_acc": checkpoint.get("val_acc", None),
            "val_loss": checkpoint.get("val_loss", None),
            "test_loss": test_loss,
            "test_acc": test_acc,
        })

    print("\n===== Summary =====")
    for test_summary in summary:
        print(f"Experiment: {test_summary['experiment']}")
        print(f"Epoch: {test_summary['epoch']}, "
              f"Val acc: {test_summary['val_acc']*100:.2f}%, Val loss: {test_summary['val_loss']:.4f}")
        print(f"Test acc: {test_summary['test_acc']*100:.2f}%, Test loss: {test_summary['test_loss']:.4f}\n")


if __name__ == "__main__":
    main()
