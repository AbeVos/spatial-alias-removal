import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import json

from math import log10
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
# from torchvision.utils import save_image

from models import SRCNN, Discriminator, EDSR, PatchGAN
from dataset import Data


def train_epoch(G, D, optim_G, optim_D, train_dataloader, device='cuda:0'):
    """
    Train both generator and discriminator for a single epoch.
    Parameters
    ----------
    G, D : torch.nn.Module
        Generator and discriminator models respectively.
    optim_G, optim_D : torch.optim.Optimizer
        Optimizers for both the models. Using Adam is recommended.
    train_dataloader : torch.utils.data.Dataloader
        Dataloader of real images to train the discriminator on.
    device : str, optional
        Device to train the models on.
    batch_size : int, optional
        Number of samples per batch.
    Returns
    -------
    tuple of float
        Tuple containing the mean loss values for the generator and
        discriminator respectively.
    """
    content_criterion = nn.MSELoss()

    mean_loss_G = []
    mean_loss_D = []
    mean_psnr = []
    criterion = nn.BCELoss()

    for sample in train_dataloader:
        lores_batch = sample['x'].to(device).float()
        hires_batch = sample['y'].to(device).float()
        print(lores_batch.shape, hires_batch.shape)

        ones = torch.ones((len(lores_batch), 1)).to(device).float()
        zeros = torch.zeros((len(lores_batch), 1)).to(device).float()

        # Train the discriminator.
        D.train()
        G.eval()
        optim_D.zero_grad()

        sures_batch = G(lores_batch)
        disc_sures = D(sures_batch.detach())
        disc_hires = D(hires_batch)

        loss = criterion(disc_hires, ones) + criterion(disc_sures, zeros)

        loss.backward()
        optim_D.step()

        mean_loss_D.append(loss.item())

        # Train the generator.
        D.eval()
        G.train()
        optim_G.zero_grad()

        sures_batch = G(lores_batch)
        disc_sures = D(sures_batch)

        content_loss = content_criterion(sures_batch, hires_batch)
        loss_G = criterion(disc_sures, ones)

        percept_loss = content_loss + loss_G
        percept_loss.backward()
        optim_G.step()

        mean_psnr += [10 * log10(1 / content_loss.item())]
        mean_loss_G.append(percept_loss.item())

    return np.mean(mean_loss_G), np.mean(mean_loss_D), np.mean(mean_psnr)


def eval_epoch(G, D, test_dataloader, device='cuda:0'):
    content_criterion = nn.MSELoss()
    mean_loss_G = []
    mean_loss_D = []
    mean_psnr = []
    criterion = nn.BCELoss()

    for sample in test_dataloader:
        lores_batch = sample['x'].to(device).float()
        hires_batch = sample['y'].to(device).float()

        ones = torch.ones((len(lores_batch), 1)).to(device).float()
        zeros = torch.zeros((len(lores_batch), 1)).to(device).float()

        # Train the discriminator.
        D.eval()
        G.eval()

        sures_batch = G(lores_batch)

        disc_sures = D(sures_batch.detach())
        disc_hires = D(hires_batch)

        loss = criterion(disc_hires, ones) + criterion(disc_sures, zeros)
        mean_loss_D.append(loss.item())
        content_loss = content_criterion(sures_batch, hires_batch)
        loss_G = criterion(disc_sures, ones)
        percept_loss = content_loss + loss_G

        mean_psnr += [10 * log10(1 / content_loss.item())]
        mean_loss_G.append(percept_loss.item())

    return np.mean(mean_loss_G), np.mean(mean_loss_D), np.mean(mean_psnr)


def plot_samples(generator, dataloader, epoch, device='cuda:0', results_directory="images"):
    """
    Plot a number of low- and high-resolution samples and the superresolution
    sample obtained from the lr image.
    """
    sample = next(iter(dataloader))

    lores_batch = sample['x'].to(device).float()
    hires_batch = sample['y'].to(device).float()

    generator.eval()

    sures_batch = generator(lores_batch)

    num_cols = 6
    num_rows = dataloader.batch_size

    plt.figure(figsize=(9, 3 * dataloader.batch_size))

    for idx, (lores, superres, hires) \
            in enumerate(zip(lores_batch, sures_batch, hires_batch)):
        plt.subplot(num_rows, num_cols, num_cols*idx+1)

        if idx == 0:
            plt.title("LR")

        plt.imshow(lores.squeeze().detach().cpu(),
                   interpolation='none', cmap='gray')
        plt.axis('off')

        plt.subplot(num_rows, num_cols, num_cols*idx+2)
        if idx == 0:
            plt.title("Superresolution")

        plt.imshow(superres.squeeze().detach().cpu(),
                   interpolation='none', cmap='gray')
        plt.axis('off')

        plt.subplot(num_rows, num_cols, num_cols*idx+3)
        if idx == 0:
            plt.title("HR")

        plt.imshow(hires.squeeze().detach().cpu(),
                   interpolation='none', cmap='gray')
        plt.axis('off')

        # Transformed
        plt.subplot(num_rows, num_cols, num_cols*idx+4)
        if idx == 0:
            plt.title("LR fk")

        lores = torch.nn.functional.interpolate(
            lores.unsqueeze(0), size=(251, 121))
        lores = torch.rfft(lores, 2, normalized=True)
        lores = lores.pow(2).sum(-1).sqrt()
        plt.imshow(lores.squeeze().detach().cpu(),
                   interpolation='none')
        plt.axis('off')

        plt.subplot(num_rows, num_cols, num_cols*idx+5)
        if idx == 0:
            plt.title("Superresolution fk")

        superres = torch.rfft(superres, 2, normalized=True)
        superres = superres.pow(2).sum(-1).sqrt()
        plt.imshow(superres.squeeze().detach().cpu(),
                   interpolation='none')
        plt.axis('off')

        plt.subplot(num_rows, num_cols, num_cols*idx+6)
        if idx == 0:
            plt.title("HR fk")

        hires = torch.rfft(hires, 2, normalized=True)
        hires = hires.pow(2).sum(-1).sqrt()
        plt.imshow(hires.squeeze().detach().cpu(),
                   interpolation='none')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"{results_directory}/gan_samples_{epoch:04d}.png")
    plt.close()


def save_loss_plot(loss_g, loss_d, directory, is_val=False):
    plt.figure()
    plt.plot(loss_d, label="Discriminator loss")
    plt.plot(loss_g, label="Generator loss")
    plt.legend()
    if is_val:
        plt.savefig("{}/gan_loss_val.png".format(directory))
    else:
        plt.savefig("{}/gan_loss.png".format(directory))
    plt.close()


def main():
    #creating new image directory
    results_directory = 'results_{}'.format(args.experiment_num)
    os.makedirs(results_directory, exist_ok=True)
    #saving arguments for reproducibility
    with open('{}/arguments.txt'.format(results_directory), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # For plotting
    plt.rcParams["figure.figsize"] = (10, 10)

    device = torch.device(args.device)


    # Determining the input data parameters and source
    if args.is_big_data:
        directory = "Data_big/"
        if args.is_fk_data:
            output_dim = [127, 154]
            filename_x = 'data_fk_20_big'
            filename_y = 'data_fk_10_big'
        else:
            output_dim = [251, 301]
            filename_x = 'data_20_big'
            filename_y = 'data_10_big'
    else:
        directory = "Data/"
        if args.is_fk_data:
            output_dim = [127, 62]
            filename_x = 'data_fk_25'
            filename_y = 'data_fk_125'
        else:
            output_dim = [251, 121]
            filename_x = 'data_25'
            filename_y = 'data_125'

    # Getting the data set and standartising it
    dataset = Data(
        filename_x=filename_x, filename_y=filename_y, directory=directory,
        transforms=transforms.Compose([
            transforms.ToTensor()
            # this is the actual statistic, not the 0,1
            # transforms.Normalize(torch.tensor(-4.4713e-07).float(),
            #                      torch.tensor(0.1018).float())
        ]))

    # Train test split
    test_size = round(len(dataset)*args.test_percentage)
    train_size = len(dataset) - test_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    # Dataloaders
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size,
                                  drop_last=True, shuffle = True)
    test_dataloader = DataLoader(test_data, batch_size=2, shuffle =True)

    # Init generator model.
    if args.model == "SRCNN":
        generator = SRCNN(input_dim=(251,61) , output_dim=output_dim).to(device)
    elif args.model == "EDSR":
        generator = EDSR(
            n_resblocks=args.num_res_blocks, output_dim=output_dim,
            latent_dim=args.latent_dim).to(device)

    # Init discriminator model.
    discriminator = Discriminator(args.is_big_data).to(device)
    # discriminator = PatchGAN().to(device)

    # Optimisers
    optim_G = optim.Adam(generator.parameters(), lr=args.lr)
    optim_D = optim.Adam(discriminator.parameters(), lr=args.lr)

    scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optim_G, patience=args.scheduler_patience, verbose=True)
    scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optim_D, patience=args.scheduler_patience, verbose=True)

    plot_G = []
    plot_D = []
    plot_G_val = []
    plot_D_val = []

    for epoch in range(args.n_epochs):
        loss_G, loss_D, mean_psnr = train_epoch(
            generator, discriminator, optim_G, optim_D, train_dataloader,
            device)

        # Report model performance.
        print(f"Epoch: {epoch}, G: {loss_G}, D: {loss_D}, PSNR: {mean_psnr}")
        #evaluation
        if epoch % args.eval_interval == 0:
            loss_G_val, loss_D_val, mean_psnr_val = eval_epoch(
                generator, discriminator, test_dataloader, device)
            print(f"Validation on epoch: {epoch}, G: {loss_G_val}, "
                  f"D: {loss_D_val}, PSNR: {mean_psnr_val}")

            # Schedule based on psnr or based on separate losses.


            if args.is_psnr_step:
                scheduler_g.step(mean_psnr_val)
                scheduler_d.step(mean_psnr_val)
            else:
                scheduler_g.step(loss_G_val)
                scheduler_d.step(loss_D_val)
            plot_G_val.append(loss_G_val)
            plot_D_val.append(loss_D_val)
        #plotting
        if epoch % args.save_interval == 0:
            plot_samples(generator, test_dataloader, epoch, device=device, results_directory = results_directory)

        plot_D.append(loss_D)
        plot_G.append(loss_G)


    # Save final plot of D and G losses.
    save_loss_plot(plot_G, plot_D, results_directory)
    save_loss_plot(plot_G_val, plot_D_val, results_directory, is_val=True)

    torch.save(generator, '{}/generator'.format(results_directory))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=1000,
                        help="number of epochs")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="batch size")
    parser.add_argument('--lr', type=float, default=0.002,
                        help="learning rate")
    parser.add_argument('--latent_dim', type=int, default=128,
                        help="dimensionality of the latent space, only "
                        "relevant for EDSR")
    parser.add_argument('--num_res_blocks', type=int, default=8,
                        help="Number of resblocks in model, only relevant "
                        "for EDSR")
    parser.add_argument('--model', type=str, default="EDSR",
                        help="Model type. EDSR or SRCNN")
    parser.add_argument('--is_big_data', action='store_true',
                        help="Is big version data")
    parser.add_argument('--is_fk_data', action='store_true',
                        help="Is fourier data")
    parser.add_argument('--test_percentage', type=float, default=0.1,
                        help="Size of the test set")
    parser.add_argument('--save_interval', type=int, default=10,
                        help="Save images every SAVE_INTERVAL epochs")
    parser.add_argument('--eval_interval', type=int, default=2,
                        help="evaluate on test set ever eval_interval epochs")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--scheduler_patience', type=int, default="10",
                        help="How many epochs of no improvement to consider Plateau")
    parser.add_argument('--is_psnr_step', type=bool, default=False,
                        help="Use PSNR for scheduler or separate losses")
    parser.add_argument('--experiment_num', type=int, default=1,
                        help="Id of the experiment running")





    args = parser.parse_args()

    main()
