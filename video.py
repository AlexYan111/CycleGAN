import torch
import torch.optim as optim
import config
from discriminator import Discriminator
from generator import Generator
from utils import load_checkpoint, tensor2im

import cv2
import numpy as np

webcam = cv2.VideoCapture('inpy.mp4')

if not webcam.isOpened():
    raise IOError("Cannot open webcam")

disc_H = Discriminator(in_channels=3).to(config.DEVICE)
disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

opt_gen = optim.Adam(
    list(gen_Z.parameters()) + list(gen_H.parameters()),
    lr=config.LEARNING_RATE,
    betas=(0.5, 0.999),
)

load_checkpoint(
    config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE,
)

out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (512,512))

while True:
    ret, original_fram = webcam.read()

    original_fram = cv2.resize(original_fram, (512,512), interpolation=cv2.INTER_AREA)
    original_fram = cv2.cvtColor(original_fram, cv2.COLOR_BGR2RGB)

    original_fram = np.array([original_fram])
    original_fram = original_fram.transpose([0,3,1,2])
        
    original_fram = torch.FloatTensor(original_fram)

    original_fram = original_fram.to(config.DEVICE)

    filtered_frame = gen_H(original_fram)
    filtered_frame = tensor2im(filtered_frame)
    filtered_frame = cv2.cvtColor(np.array(filtered_frame), cv2.COLOR_BGR2RGB)  
    filtered_frame = cv2.resize(filtered_frame, (512, 512))      

    out.write(filtered_frame)

    cv2.imshow('Input', filtered_frame)
    c = cv2.waitKey(1)
    if c == 27:
        break


webcam.release()
cv2.destroyAllWindows()