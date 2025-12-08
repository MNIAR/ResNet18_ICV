import os

from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
import torchvision.transforms as t
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data_dir, data_type="train", size=(224, 224), is_augment=False,
                 transform=None, target_transform=None):
        super().__init__()

        self.data_dir = data_dir
        self.data_type = data_type
        self.size = size
        self.is_augment = is_augment
        self.transform = transform
        self.target_transform = target_transform

        self.image_names, self.labels = self.__process_data()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        label = self.labels[idx]

        label_map = {0: 'Chorionic_villi', 1: 'Decidual_tissue', 2: 'Hemorrhage', 3: 'Trophoblastic_tissue'}

        image_path = os.path.join(self.data_dir, self.data_type, label_map[label], image_name)
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        if self.is_augment:
            ## save the augmented image
            saved_image_path = os.path.join(self.data_dir, self.data_type, label_map[label],
                                            image_name.split('.')[0] + '_augmented.jpg')
            while os.path.exists(saved_image_path):
                random_number = np.random.randint(0, 100)
                saved_image_path = saved_image_path.split('.')[0] + f'_{random_number}.jpg'

            ## convert the tensor to numpy array
            ## apply random rotation (0, 90) degrees to PIL image
            image = t.RandomRotation(degrees=(0, 90))(image)
            image = t.RandomHorizontalFlip(p=1)(image)
            image.save(saved_image_path)

        return image, torch.tensor(label).long()

    def __process_data(self):
        if isinstance(self.data_dir, tuple):
            self.data_dir = os.path.join(*self.data_dir)

        chorionic_villi = os.listdir(os.path.join(self.data_dir, self.data_type, 'Chorionic_villi'))
        decidual_tissue = os.listdir(os.path.join(self.data_dir, self.data_type, 'Decidual_tissue'))
        hemorrhage = os.listdir(os.path.join(self.data_dir, self.data_type, 'Hemorrhage'))
        trophoblastic_tissue = os.listdir(os.path.join(self.data_dir, self.data_type, 'Trophoblastic_tissue'))

        combined_images = chorionic_villi + decidual_tissue + hemorrhage + trophoblastic_tissue
        labels = (
                [0] * len(chorionic_villi)
                + [1] * len(decidual_tissue)
                + [2] * len(hemorrhage)
                + [3] * len(trophoblastic_tissue)
        )

        return combined_images, labels

    def plot_grid_images(self, x, y, batch_size):
        rows = cols = int(batch_size ** 0.5)
        fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
        axes = axes.flatten()

        for i in range(batch_size):
            axes[i].imshow(x[i].permute(1, 2, 0))
            axes[i].set_title(f"Label: {y[i]}")
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()