import sys
import torch
import torch.nn as nn
import torch.optim as optim
import config
from data_preprocess import HorseZebraDataset
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator

def train(disc_H, disc_Z, Gen_Zebra, Gen_Horse, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    Horse_reals = 0
    Horse_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            #Loss of Discriminator of zebra
            fake_zebra = Gen_Zebra(horse)
            Dis_Zebra_real = disc_Z(zebra)
            Dis_Zebra_fake = disc_Z(fake_zebra.detach())
            Dis_Zebra_real_loss = mse(Dis_Zebra_real, torch.ones_like(Dis_Zebra_real))
            Dis_Zebra_fake_loss = mse(Dis_Zebra_fake, torch.zeros_like(Dis_Zebra_fake))
            Dis_Zebra_loss = Dis_Zebra_real_loss + Dis_Zebra_fake_loss

            #Loss of Discriminator of Horse
            fake_horse = Gen_Horse(zebra)
            Dis_Horse_real = disc_H(horse)
            Dis_Horse_fake = disc_H(fake_horse.detach())
            Horse_reals += Dis_Horse_real.mean().item()
            Horse_fakes += Dis_Horse_fake.mean().item()
            Dis_Horse_real_loss = mse(Dis_Horse_real, torch.ones_like(Dis_Horse_real))
            Dis_Horse_fake_loss = mse(Dis_Horse_fake, torch.zeros_like(Dis_Horse_fake))
            Dis_Horse_loss = Dis_Horse_real_loss + Dis_Horse_fake_loss

            #Loss of two Discriminators
            Dis_loss = (Dis_Horse_loss + Dis_Zebra_loss)/2

        opt_disc.zero_grad()
        d_scaler.scale(Dis_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            Dis_Horse_fake = disc_H(fake_horse)
            Dis_Zebra_fake = disc_Z(fake_zebra)
            loss_Gen_Horse = mse(Dis_Horse_fake, torch.ones_like(Dis_Horse_fake))
            loss_Gen_Zebra = mse(Dis_Zebra_fake, torch.ones_like(Dis_Zebra_fake))

            # cycle loss
            cycle_zebra = Gen_Zebra(fake_horse)
            cycle_horse = Gen_Horse(fake_zebra)
            cycle_zebra_loss = l1(zebra, cycle_zebra)
            cycle_horse_loss = l1(horse, cycle_horse)

            # identity loss
            identity_zebra = Gen_Zebra(zebra)
            identity_horse = Gen_Horse(horse)
            identity_zebra_loss = l1(zebra, identity_zebra)
            identity_horse_loss = l1(horse, identity_horse)

            #Loss = adversarial loss for both generators + cycle loss + identity loss
            Gen_loss = (
                loss_Gen_Zebra + loss_Gen_Horse 
                + cycle_zebra_loss * config.LAMBDA_CYCLE + cycle_horse_loss * config.LAMBDA_CYCLE 
                + identity_horse_loss * config.LAMBDA_IDENTITY + identity_zebra_loss * config.LAMBDA_IDENTITY
            )

        #backpropagation
        opt_gen.zero_grad()
        g_scaler.scale(Gen_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        loop.set_postfix(Horse_real=Horse_reals/(idx+1), Horse_fake=Horse_fakes/(idx+1))

def test(Gen_Zebra, Gen_Horse, val_loader):
    loop = tqdm(val_loader, leave=True)
    for idx, (zebra, horse) in enumerate(loop):
        #load original horse and zebra
        z = zebra.to(config.DEVICE)
        h = horse.to(config.DEVICE)

        #Get fake horse and zebra
        fake_h = Gen_Horse(z)
        fake_z = Gen_Zebra(h)

        #save fake horse and fake zebra
        save_image(fake_h*0.5+0.5, f"Project7_CycleGAN/saved_images/fake_tatsuki_{idx}.png")
        save_image(fake_z*0.5+0.5, f"Project7_CycleGAN/saved_images/fake_photo_{idx}.png")


def main():
    #Get two discriminators for fake horse and two generators for fake zebras
    Dis_Horse = Discriminator(in_channels=3).to(config.DEVICE)
    Dis_Zebra = Discriminator(in_channels=3).to(config.DEVICE)
    Gen_Zebra = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    Gen_Horse = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    #Get two optimizers for discriminator and generator
    opt_gen = optim.Adam(
        list(Gen_Zebra.parameters()) + list(Gen_Horse.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    opt_disc = optim.Adam(
        list(Dis_Horse.parameters()) + list(Dis_Zebra.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    #Load two functions to compute loss
    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    #load our model
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H, Gen_Horse, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Z, Gen_Zebra, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H, Dis_Horse, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Z, Dis_Zebra, opt_disc, config.LEARNING_RATE,
        )

    #Load training data and testing data
    dataset = HorseZebraDataset(
        dir_horse=config.TRAIN_DIR+"/monet", dir_zebra=config.TRAIN_DIR+"/photos", transform=config.transforms
    )

    val_dataset = HorseZebraDataset(
       dir_horse=config.VAL_DIR+"/monet", dir_zebra=config.VAL_DIR+"/photos", transform=config.transforms
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    #Train the model
    for epoch in range(config.NUM_EPOCHS):
        #print current percentage with epoch and total num of epochs
        print('\r', )
        print("percentage {}".format(epoch / config.NUM_EPOCHS))

        #train model
        train(Dis_Horse, Dis_Zebra, Gen_Zebra, Gen_Horse, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

        #save trained model
        if config.SAVE_MODEL:
            save_checkpoint(Dis_Horse, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
            save_checkpoint(Dis_Zebra, opt_disc, filename=config.CHECKPOINT_CRITIC_Z)
            save_checkpoint(Gen_Horse, opt_gen, filename=config.CHECKPOINT_Gen_Horse)
            save_checkpoint(Gen_Zebra, opt_gen, filename=config.CHECKPOINT_Gen_Zebra)
    
    #Test model after training 
    test(Gen_Zebra, Gen_Horse, val_loader)

if __name__ == "__main__":
    main()