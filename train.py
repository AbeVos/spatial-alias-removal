import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from math import log10
from statistics import mean
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from dataset import Data
from models import SRCNN, Discriminator, EDSR, PatchGAN


def split_dataset(dataset, test_percentage=0.1):
    """
    Split a dataset in a train and test set.

    Parameters
    ----------
    dataset : dataset.Data
        Custom dataset object.
    test_percentage : float, optional
        Percentage of the data to be assigned to the test set.
    """
    test_size = round(len(dataset) * test_percentage)
    train_size = len(dataset) - test_size
    return random_split(dataset, [train_size, test_size])


def iter_epoch(G, D, optim_G, optim_D, dataset, device='cuda:0',
                batch_size=64, eval=False):
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
    eval : bool, optional
        If `True`, model parameters are not updated

    Returns
    -------
    tuple of float
        Tuple containing the mean loss values for the generator and
        discriminator, and the mean PSNR respectively.
    """
    def train_update(lores_batch, hires_batch):
        """
        Update the model over a single minibatch.
        """
        # Train the discriminator.
        D.train()
        G.eval()

        sures_batch = G(lores_batch)
        disc_sures = D(sures_batch.detach())
        disc_hires = D(hires_batch)

        loss_D = criterion(disc_hires, ones) + criterion(disc_sures, zeros)

        loss_D.backward()
        optim_D.step()
        optim_D.zero_grad()

        # Train the generator.
        D.eval()
        G.train()

        sures_batch = G(lores_batch)
        disc_sures = D(sures_batch)

        content_loss = content_criterion(sures_batch, hires_batch)
        loss_G = criterion(disc_sures, ones)
        percept_loss = content_loss + loss_G

        percept_loss.backward()
        optim_G.step()
        optim_G.zero_grad()

        psnr = 10 * log10(1 / content_loss.item())

        return loss_D.item(), percept_loss.item(), psnr

    def eval_update(lores_batch, hires_batch):
        """
        Evaluate the model for a single mini-batch.
        """
        D.eval()
        G.eval()

        sures_batch = G(lores_batch)
        disc_sures = D(sures_batch)
        disc_hires = D(hires_batch)

        loss_D = criterion(disc_hires, ones) + criterion(disc_sures, zeros)
        loss_G = criterion(disc_sures, ones)

        content_loss = content_criterion(sures_batch, hires_batch)
        percept_loss = content_loss + loss_G

        psnr = 10 * log10(1 / content_loss.item())

        return loss_D.item(), percept_loss.item(), psnr

    dataloader = DataLoader(
        dataset, batch_size=batch_size, drop_last=(not eval), shuffle=True)

    mean_loss_G = []
    mean_loss_D = []
    mean_psnr = []

    content_criterion = nn.MSELoss()
    criterion = nn.BCELoss()

    for sample in dataloader:
        lores_batch = sample['x'].to(device).float()
        hires_batch = sample['y'].to(device).float()

        # Create label tensors.
        ones = torch.ones((len(lores_batch), 1)).to(device).float()
        zeros = torch.zeros((len(lores_batch), 1)).to(device).float()

        if not eval:
            loss_D, loss_G, psnr = train_update(lores_batch, hires_batch)
        else:
            loss_D, loss_G, psnr = eval_update(lores_batch, hires_batch)

        mean_loss_G.append(loss_G)
        mean_loss_D.append(loss_D)
        mean_psnr.append(psnr)

    return mean(mean_loss_G), mean(mean_loss_D), mean(mean_psnr)


def plot_samples(generator, dataset, epoch, device='cuda', directory='image'):
    """
    Plot data samples, their superresolution and the corresponding fk
    transforms.
    """
    def add_subplot(plt, image, i, idx, title=None, cmap='viridis'):
        plt.subplot(num_rows, num_cols, num_cols * idx + i)

        if idx == 0:
            plt.title(title)

        plt.imshow(image.squeeze().detach().cpu(),
                   interpolation='none', cmap=cmap)
        plt.axis('off')

    def transform_fk(image):
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0), size=(251, 121))
        image_fk = torch.rfft(image, 2, normalized=True)
        image_fk = image_fk.pow(2).sum(-1).sqrt()

        return image_fk

    dataloader = DataLoader(dataset, shuffle=False, batch_size=2)
    sample = next(iter(dataloader))

    lores_batch = sample['x'].to(device).float()
    hires_batch = sample['y'].to(device).float()

    generator.eval()

    sures_batch = generator(lores_batch)

    num_cols = 6
    num_rows = dataloader.batch_size

    plt.figure(figsize=(9, 3 * num_rows))

    for idx, (lores, sures, hires) \
            in enumerate(zip(lores_batch, sures_batch, hires_batch)):
        # Plot images.
        add_subplot(plt, lores, 1, idx, "LR", cmap='gray')
        add_subplot(plt, sures, 2, idx, "SR", cmap='gray')
        add_subplot(plt, hires, 3, idx, "HR", cmap='gray')

        # Plot transformed images.
        add_subplot(plt, transform_fk(lores), 4, idx, "LR fk")
        add_subplot(plt, transform_fk(sures), 5, idx, "SR fk")
        add_subplot(plt, transform_fk(hires), 6, idx, "HR fk")

    plt.tight_layout()
    plt.savefig(os.path.join(directory, 'samples_{epoch}.png'))
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


def main(args):
    # Create directories.
    results_directory = f'results/result_{args.experiment_num}'
    os.makedirs('images', exist_ok=True)
    os.makedirs(results_directory, exist_ok=True)

    # Save arguments for experiment reproducibility.
    with open(os.path.join(results_directory, 'arguments.txt'), 'w') as file:
        json.dump(args.__dict__, file, indent=2)

    # Set size for plots.
    plt.rcParams['figure.figsize'] = (10, 10)

    # Select the device to train the model on.
    device = torch.device(args.device)

    # Load the dataset.
    dataset = Data(
        args.filename_x, args.filename_y, args.data_root,
        transforms=transforms.Compose([
            transforms.ToTensor(),
        ])
    )

    train_data, test_data = split_dataset(dataset, args.test_percentage)

    # Initialize generator model.
    if args.model == 'SRCNN':
        generator = SRCNN(dataset.input_dim, dataset.output_dim).to(device)
    elif args.model == 'EDSR':
        generator = EDSR(
            args.latent_dim, args.num_res_blocks,
            output_dim=dataset.output_dim).to(device)

    # Initialize the discriminator model.
    # TODO: fix hardcoded size of discriminator.
    discriminator = Discriminator().to(device)

    # Optimizers
    optim_G = optim.Adam(generator.parameters(), lr=args.lr)
    optim_D = optim.Adam(discriminator.parameters(), lr=args.lr)

    scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optim_G, patience=args.scheduler_patience, verbose=True)
    scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optim_D, patience=args.scheduler_patience, verbose=True)

    # Initialize empty lists for plotting.
    plot_G = []
    plot_D = []
    plot_G_val = []
    plot_D_val = []

    for epoch in range(args.n_epochs):
        # Train model for one epoch.
        loss_G, loss_D, mean_psnr = iter_epoch(
            generator, discriminator, optim_G, optim_D, train_data, device,
            batch_size=args.batch_size)

        # Report model performance.
        print(f"Epoch: {epoch}, G: {loss_G}, D: {loss_D}, PSNR: {mean_psnr}")
        plot_D.append(loss_D)
        plot_G.append(loss_G)

        # Model evaluation.
        if epoch % args.eval_interval == 0:
            loss_G_val, loss_D_val, mean_psnr_val = iter_epoch(
                generator, discriminator, None, None, test_data, device,
                batch_size=args.batch_size, eval=True)
            print(f"Validation on epoch: {epoch}, G: {loss_G_val}, "
                  f"D: {loss_D_val}, PSNR: {mean_psnr_val}")

            plot_G_val.append(loss_G_val)
            plot_D_val.append(loss_D_val)

            # Update scheduler based on PSNR or separate model losses.
            if args.is_psnr_step:
                scheduler_g.step(mean_psnr_val)
                scheduler_d.step(mean_psnr_val)
            else:
                scheduler_g.step(loss_G_val)
                scheduler_d.step(loss_D_val)

        # Plot results.
        if epoch % args.save_interval == 0:
            plot_samples(generator, test_data, epoch, device,
                         results_directory)

        save_loss_plot(plot_G, plot_D, results_directory)

    # Save final loss plots.
    save_loss_plot(plot_G, plot_D, results_directory)
    save_loss_plot(plot_G_val, plot_D_val, results_directory, is_val=True)

    # Save the trained generator model.
    torch.save(generator, os.path.join(results_directory, 'generator.pth'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data arguments.
    data_group = parser.add_argument_group('Data')
    data_group.add_argument(
        '--data_root', type=str, default='Data',
        help="Root directory of the data.")
    data_group.add_argument(
        '--filename_x', type=str, default='data_25',
        help="Name of the data input file (without the '.mat' extension).")
    data_group.add_argument(
        '--filename_y', type=str, default='data_125',
        help="Name of the data output file (without the '.mat' extension).")
    data_group.add_argument(
        '--test_percentage', type=float, default=0.1,
        help="Size of the test set")

    # Model arguments.
    model_group = parser.add_argument_group('Model')
    model_group.add_argument(
        '--model', type=str, default="EDSR",
        help="Model type. EDSR or SRCNN")
    model_group.add_argument(
        '--latent_dim', type=int, default=128,
        help="dimensionality of the latent space, only relevant for EDSR")
    model_group.add_argument(
        '--num_res_blocks', type=int, default=8,
        help="Number of resblocks in model, only relevant for EDSR")

    # Training arguments.
    training_group = parser.add_argument_group('Training')
    training_group.add_argument(
        '--n_epochs', type=int, default=1000,
        help="number of epochs")
    training_group.add_argument(
        '--batch_size', type=int, default=8,
        help="batch size")
    training_group.add_argument(
        '--lr', type=float, default=0.002,
        help="learning rate")
    training_group.add_argument(
        '--scheduler_patience', type=int, default="10",
        help="How many epochs of no improvement to consider Plateau")
    training_group.add_argument(
        '--is_psnr_step', action='store_true',
        help="Use PSNR for scheduler or separate losses")

    # Misc arguments.
    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument(
        '--eval_interval', type=int, default=10,
        help="evaluate on test set ever eval_interval epochs")
    misc_group.add_argument(
        '--save_interval', type=int, default=10,
        help="Save images every SAVE_INTERVAL epochs")
    misc_group.add_argument(
        '--device', type=str, default="cpu",
        help="Training device 'cpu' or 'cuda:0'")
    misc_group.add_argument(
        '--experiment_num', type=int, default=1,
        help="Id of the experiment running")

    args = parser.parse_args()

    main(args)
