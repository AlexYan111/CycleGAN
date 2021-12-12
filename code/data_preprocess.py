from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class HorseZebraDataset(Dataset):
    def __init__(self, dir_zebra, dir_horse, transform=None):
        self.dir_zebra = dir_zebra
        self.dir_horse = dir_horse
        self.transform = transform

        self.zebra_images = os.listdir(dir_zebra)
        self.horse_images = os.listdir(dir_horse)
        self.length_dataset = max(len(self.zebra_images), len(self.horse_images)) # 1000, 1500
        self.zebra_len = len(self.zebra_images)
        self.horse_len = len(self.horse_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        zebra_image = self.zebra_images[index % self.zebra_len]
        horse_image = self.horse_images[index % self.horse_len]

        zebra_path = os.path.join(self.dir_zebra, zebra_image)
        horse_path = os.path.join(self.dir_horse, horse_image)

        zebra_image = np.array(Image.open(zebra_path).convert("RGB"))
        horse_image = np.array(Image.open(horse_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=zebra_image, image0=horse_image)
            zebra_image = augmentations["image"]
            horse_image = augmentations["image0"]

        return zebra_image, horse_image
