import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt


def plot_heatmap(labels, preds, num_classes, exp_name):
    
    label_map = {0: 'Chorionic_villi', 1: 'Decidual_tissue', 2: 'Hemorrhage', 3: 'Trophoblastic_tissue'}
    
    cm = confusion_matrix(labels, preds)
    precision = precision_score(labels, preds, average=None, zero_division=0)
    recall = recall_score(labels, preds, average=None, zero_division=0)
    f1 = f1_score(labels, preds, average=None, zero_division=0)

    class_acc = cm.diagonal() / cm.sum(axis=1)

    metrics = []
    for i in range(num_classes):
        metrics.append([
            class_acc[i],
            precision[i],
            recall[i],
            f1[i]
        ])

    metrics = torch.tensor(metrics)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        metrics.numpy(),
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=["Acc", "Precision", "Recall", "F1"],
        yticklabels=[label_map[i] for i in range(num_classes)]
    )
    plt.title(f"Metrics Heatmap ({exp_name})")
    plt.tight_layout()
    plt.savefig(f"metrics_heatmap_{exp_name}.png")
    plt.close()
