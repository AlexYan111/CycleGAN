import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "Project7_CycleGAN/data/train"
VAL_DIR = "Project7_CycleGAN/data/val"
SAVE_DIR = "Project7_CycleGAN/save_images"
BATCH_SIZE = 3
LEARNING_RATE = 0.0002
LAMBDA_IDENTITY = 0.5
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 120
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN_H = "gen_photo2monet.pth.tar"
CHECKPOINT_GEN_Z = "gen_monet2photo.pth.tar"
CHECKPOINT_CRITIC_H = "critich.pth.tar"
CHECKPOINT_CRITIC_Z = "criticz.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)

print(torch.cuda.is_available())
