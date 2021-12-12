import torch
from data_preprocess import HorseZebraDataset
import sys
from utils import save_checkpoint, load_checkpoint, tensor2im
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator

import cv2
import numpy as np

webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    raise IOError("Cannot open webcam")

disc_H = Discriminator(in_channels=3).to(config.DEVICE)
disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
 
opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

opt_gen = optim.Adam(
    list(gen_Z.parameters()) + list(gen_H.parameters()),
    lr=config.LEARNING_RATE,
    betas=(0.5, 0.999),
)

load_checkpoint(
    config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE,
)

load_checkpoint(
    config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE,
)

#text set up
# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
# org 
org = (0, 25) 
# fontScale 
fontScale = 1
# Blue color in BGR 
color = (255, 255, 255) 
# Line thickness of 2 px 
thickness = 2

while True:
    ret, frame = webcam.read()
    if ret == True:
        frame = cv2.resize(frame, (512,512), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = np.array([frame])
        frame = frame.transpose([0,3,1,2])
        
        
        #now shape is batchsize * channels * h * w
        
        frame = torch.FloatTensor(frame)

        frame = frame.to(config.DEVICE)

        result_image = gen_H(frame)
        result_image = tensor2im(result_image)
        #result_image = tensor2im(frame)
        result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_BGR2RGB)  
        result_image = cv2.resize(result_image, (512, 512))      
        #result_image = cv2.putText(result_image, str(opt.name)[6:-11], org, font,fontScale, color, thickness, cv2.LINE_AA)   


        cv2.imshow('Input', result_image)
        c = cv2.waitKey(1)
        if c == 27:
            break
    else:
        print("frame is Empty")
        break


webcam.release()
cv2.destroyAllWindows()
