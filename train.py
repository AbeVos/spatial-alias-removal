import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from collections import defaultdict
from math import log10
from statistics import mean
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from dataset import Data
from models import SRCNN, Discriminator, EDSR, PatchGAN, VDSR


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


def iter_epoch(
        models, optimizers, dataset, device='cuda:0', batch_size=64,
        eval=False, reconstruction_criterion=nn.MSELoss(),
        use_gan=True, use_fk_loss=False, use_noisy_labels = False):
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
                    batch_size=64, eval=False,
    reconstruction_criterion: loss used to evaluate the reconstruction quality
        options: nn.MSELoss(), nn.L1Loss(), None (if used, only GAN loss is
        counted)
    is_gan: bool
        If 'True', GAN loss is used, else just reconstruction loss
    is_fk_loss: bool
        If 'True', loss is evaluated in the fk space, else loss is evaluated
        directly

    Returns
    -------
    dict
        Dictionary containing the mean loss values for the generator and
        discriminator, and the mean PSNR respectively.
    """
    def update_discriminator(D, lores_batch, hires_batch,
                             use_fk_loss=False):
        """Update the discriminator over a single minibatch."""
        if eval:
            D.eval()
        else:
            D.train()
        G.eval()

        if use_fk_loss:
            hires_batch = transform_fk(hires_batch, output_dim, True)

        # Generate superresolution and transform.
        sures_batch = G(lores_batch)

        if use_fk_loss:
            sures_batch = transform_fk(sures_batch, output_dim, is_batch=True)

        disc_sures = D(sures_batch.detach())
        disc_hires = D(hires_batch)

        loss_D = criterion(disc_hires, ones) + criterion(disc_sures, zeros)

        if not eval:
            loss_D.backward()
            optim_D.step()
            optim_D.zero_grad()

        return loss_D.item()

    def update_generator(lores_batch, hires_batch, use_gan):
        """Update the generator over a single minibatch."""
        D.eval()
        if D_fk is not None:
            D_fk.eval()

        if eval:
            G.eval()
        else:
            G.train()

        # Generate superresolution and transform.
        sures_batch = G(lores_batch)

        if use_fk_loss:
            hires_fk_batch = transform_fk(
                hires_batch, output_dim, is_batch=True)
            sures_fk_batch = transform_fk(
                sures_batch, output_dim, is_batch=True)

        # Initialize losses.
        gen_loss = 0
        gen_fk_loss = 0
        rec_loss = 0
        rec_fk_loss = 0

        if use_gan:
            disc_sures = D(sures_batch)
            gen_loss = criterion(disc_sures, ones)

            if use_fk_loss:
                disc_fk_sures = D_fk(sures_fk_batch)
                gen_fk_loss = criterion(disc_fk_sures, ones)
        elif content_criterion is None:
            raise Exception("Cannot use None reconstruction loss without GAN")

        if content_criterion is not None:
            rec_loss = content_criterion(sures_batch, hires_batch)

            if use_fk_loss:
                rec_fk_loss = content_criterion(sures_fk_batch, hires_fk_batch)

        loss_G = gen_loss + gen_fk_loss + rec_loss + rec_fk_loss

        if not eval:
            loss_G.backward()
            optim_G.step()
            optim_G.zero_grad()

        psnr = 10 * log10(1 / nn.functional.mse_loss(sures_batch, hires_batch).item())

        return loss_G.item(), psnr

    G, D, D_fk = models
    optim_G, optim_D, optim_D_fk = optimizers

    output_dim = dataset[0]['y'].shape[1:]
    dataloader = DataLoader(
        dataset, batch_size=batch_size, drop_last=(not eval), shuffle=True)

    mean_loss_G = []
    mean_loss_D = []
    mean_loss_D_fk = []
    mean_psnr = []

    content_criterion = reconstruction_criterion
    criterion = nn.BCEWithLogitsLoss()

    for sample in dataloader:
        lores_batch = sample['x'].to(device).float()
        hires_batch = sample['y'].to(device).float()

        # Create label tensors.
        if use_noisy_labels:
            ones = torch.empty((len(lores_batch), 1)).uniform_(0.7,1.2).to(device).float()
            zeros = torch.empty((len(lores_batch), 1)).uniform_(0,0.3).to(device).float()
        else:
            ones = torch.ones((len(lores_batch), 1)).to(device).float()
            zeros = torch.zeros((len(lores_batch), 1)).to(device).float()

        loss_D = 0
        loss_D_fk = 0

        if use_gan:
            loss_D = update_discriminator(
                D, lores_batch, hires_batch, use_fk_loss=False)

            if use_fk_loss:
                loss_D_fk = update_discriminator(
                    D_fk, lores_batch, hires_batch, use_fk_loss=True)

        loss_G, psnr = update_generator(lores_batch, hires_batch, use_gan)

        mean_loss_G.append(loss_G)
        mean_loss_D.append(loss_D)
        mean_loss_D_fk.append(loss_D_fk)
        mean_psnr.append(psnr)

    return {
        'G': mean(mean_loss_G),
        'D': mean(mean_loss_D),
        'D_fk': mean(mean_loss_D_fk),
        'psnr': mean(mean_psnr)
    }


def transform_fk(image, dataset_dim, is_batch=False):
    """
    Apply the Fourier transform of an image (or batch of images) and
    compute the magnitude of its real and imaginary parts.
    """
    if not is_batch:
        image = image.unsqueeze(0)

    image = torch.nn.functional.interpolate(image, size=dataset_dim)
    image_fk = torch.rfft(image, 2, normalized=True)
    image_fk = image_fk.pow(2).sum(-1).sqrt()

    return image_fk


def plot_samples(generator, dataset, epoch, device='cuda', directory='image',
                 is_train=False):
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

    dataloader = DataLoader(dataset, shuffle=False, batch_size=2)
    sample = next(iter(dataloader))

    lores_batch = sample['x'].to(device).float()
    hires_batch = sample['y'].to(device).float()

    generator.eval()

    sures_batch = generator(lores_batch)

    num_cols = 6
    num_rows = dataloader.batch_size
    output_dim = dataset[0]['y'].shape[1:]


    plt.figure(figsize=(9, 3 * num_rows))

    for idx, (lores, sures, hires) \
            in enumerate(zip(lores_batch, sures_batch, hires_batch)):
        # Plot images.
        add_subplot(plt, lores, 1, idx, "LR", cmap='gray')
        add_subplot(plt, sures, 2, idx, "SR", cmap='gray')
        add_subplot(plt, hires, 3, idx, "HR", cmap='gray')

        # Plot transformed images.
        add_subplot(plt, transform_fk(lores, output_dim), 4, idx, "LR fk")
        add_subplot(plt, transform_fk(sures, output_dim), 5, idx, "SR fk")
        add_subplot(plt, transform_fk(hires, output_dim), 6, idx, "HR fk")

    plt.tight_layout()
    if not is_train:
        plt.savefig(os.path.join(directory, f'samples_{epoch:03d}.png'))
    else:
        plt.savefig(os.path.join(directory, f'samples_{epoch:03d}_train.png'))
    plt.close()


def save_loss_plot(loss_g, loss_d, directory, is_val=False, name=None):
    plt.figure()
    plt.plot(loss_d, label="Discriminator loss")
    plt.plot(loss_g, label="Generator loss")
    plt.legend()

    if is_val:
        if name is None:
            plt.savefig(f"{directory}/gan_loss_val.png")
        else:
            plt.savefig(f"{directory}/gan_loss_val_{name}.png")
    else:
        if name is None:
            plt.savefig(f"{directory}/gan_loss.png")
        else:
            plt.savefig(f"{directory}/gan_loss_{name}.png")

    plt.close()


def main(args):
    # Create directories if it's not  hyper-optimisation round.
    if not args.is_optimisation:
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
    # TODO : Add normalisation  transforms.Normalize(
    #   torch.tensor(-4.4713e-07).float(),
    #   torch.tensor(0.1018).float())
    dataset = Data(
        args.filename_x, args.filename_y, args.data_root,
        transforms=transforms.Compose([
            transforms.ToTensor(),
        ])
    )
    if not args.is_optimisation:
        print(f"Data sizes, input: {dataset.input_dim}, output: "
            f"{dataset.output_dim}, Fk: {dataset.output_dim_fk}")

    train_data, test_data = split_dataset(dataset, args.test_percentage)

    # Initialize generator model.
    if args.model == 'SRCNN':
        generator = SRCNN(input_dim=dataset.input_dim,
                          output_dim=dataset.output_dim).to(device)
    elif args.model == 'EDSR':
        generator = EDSR(
            args.latent_dim, args.num_res_blocks,
            output_dim=dataset.output_dim).to(device)
    elif args.model == 'VDSR':
        generator = VDSR(
            args.latent_dim, args.num_res_blocks,
            output_dim=dataset.output_dim).to(device)

    # Initialize the discriminator model.
    discriminator = Discriminator(input_dim=dataset.output_dim).to(device)

    # Optimizers
    optim_G = optim.Adam(generator.parameters(), lr=args.lr)
    optim_D = optim.Adam(discriminator.parameters(), lr=args.lr)

    scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optim_G, patience=args.scheduler_patience, verbose=True)
    scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optim_D, patience=args.scheduler_patience, verbose=True)

    # Initialize optional Fk discriminator and optimizer.
    if args.use_fk_loss and args.is_gan:
        discriminator_fk = Discriminator(
            input_dim=dataset.output_dim_fk).to(device)
        optim_D_fk = optim.Adam(discriminator_fk.parameters(), lr=args.lr)
        scheduler_d_fk = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optim_D_fk, patience=args.scheduler_patience,
            verbose=True)
    else:
        discriminator_fk = None
        optim_D_fk = None

    # losses type
    criterion_dictionary = {
        "MSE": nn.MSELoss(),
        "L1": nn.L1Loss(),
        "None": None
    }
    reconstruction_criterion = criterion_dictionary[args.criterion_type]

    # Initialize a dict of empty lists for plotting.
    plot_log = defaultdict(list)

    for epoch in range(args.n_epochs):
        # Train model for one epoch.
        loss = iter_epoch(
            (generator, discriminator, discriminator_fk),
            (optim_G, optim_D, optim_D_fk), train_data, device,
            batch_size=args.batch_size,
            reconstruction_criterion=reconstruction_criterion,
            use_gan=args.is_gan, use_fk_loss=args.use_fk_loss)

        # Report model performance.
        if  not args.is_optimisation:
            print(f"Epoch: {epoch}, G: {loss['G']}, D: {loss['D']}, "
                f"PSNR: {loss['psnr']}")
        plot_log['G'].append(loss['G'])
        plot_log['D'].append(loss['D'])

        # Model evaluation every eval_iteration and last iteration.
        if epoch % args.eval_interval == 0 or (args.is_optimisation and epoch == args.n_epochs - 1 ):
            loss_val = iter_epoch(
                (generator, discriminator, discriminator_fk),
                (None, None, None), test_data, device,
                batch_size=args.batch_size, eval=True,
                reconstruction_criterion=reconstruction_criterion,
                use_gan=args.is_gan, use_fk_loss=args.use_fk_loss, use_noisy_labels=args.is_noisy_label)
            if not args.is_optimisation:
                print(f"Validation on epoch: {epoch}, G: {loss_val['G']}, "
                       f"D: {loss_val['D']}, PSNR: {loss_val['psnr']}")

            plot_log['G_val'].append(loss_val['G'])
            plot_log['D_val'].append(loss_val['D'])
            plot_log['psnr_val'].append(loss_val['psnr'])

            # Update scheduler based on PSNR or separate model losses.
            if args.is_psnr_step:
                scheduler_g.step(loss_val['psnr'])
                scheduler_d.step(loss_val['psnr'])
            else:
                scheduler_g.step(loss_val['G'])
                scheduler_d.step(loss_val['D'])
            if not args.is_optimisation:
                save_loss_plot(plot_log['G_val'], plot_log['D_val'],
                              results_directory, is_val=True)

        if not args.is_optimisation:
            # Plot results.
            if epoch % args.save_interval == 0:
                plot_samples(generator, test_data, epoch, device,
                             results_directory)
                plot_samples(generator, train_data, epoch, device,
                             results_directory, is_train=True)

            save_loss_plot(plot_log['G'], plot_log['D'], results_directory)

    if not args.is_optimisation:
        # Save the trained generator model.
        torch.save(generator, os.path.join(results_directory, 'generator.pth'))
    if args.is_optimisation:
        return plot_log, generator, test_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data arguments.
    data_group = parser.add_argument_group('Data')

    data_group.add_argument(
        '--data_root', type=str, default='Data_big/',
        help="Root directory of the data.")
    data_group.add_argument(
        '--filename_x', '-x', type=str, default='data_20_big',
        help="Name of the low resolution data file (without the '.mat' "
        "extension).")
    data_group.add_argument(
        '--filename_y', '-y', type=str, default='data_10_big',
        help="Name of the high resolution data filee (without the '.mat' "
        "extension).")
    data_group.add_argument(
        '--test_percentage', type=float, default=0.1,
        help="Size of the test set")

    # Model arguments.
    model_group = parser.add_argument_group('Model')

    model_group.add_argument(
        '--model', type=str, default="VDSR",
        choices=['EDSR', 'SRCNN', "VDSR"],
        help="Model type.")
    model_group.add_argument(
        '--latent_dim', type=int, default=256,
        help="dimensionality of the latent space, only relevant for EDSR and VDSR")
    model_group.add_argument(
        '--num_res_blocks', type=int, default=6,
        help="Number of resblocks in model, only relevant for EDSR and VDSR")



    # Training arguments.
    training_group = parser.add_argument_group('Training')

    training_group.add_argument(
        '--n_epochs', type=int, default=100,
        help="number of epochs")
    training_group.add_argument(
        '--batch_size', type=int, default=8,
        help="batch size")
    training_group.add_argument(
        '--lr', type=float, default=0.001,
        help="learning rate")
    training_group.add_argument(
        '--scheduler_patience', type=int, default="5",
        help="How many epochs of no improvement to consider Plateau")
    training_group.add_argument(
        '--is_psnr_step', type=int, default="0",
        help="Use PSNR for scheduler or separate losses")
    training_group.add_argument(
        '--criterion_type', type=str, default="L1",
        choices=['MSE', 'L1', 'None'],
        help="Reconstruction criterion to use.")
    training_group.add_argument(
        '--is_gan',  type=int, default="0",
        help="If set, use GAN loss.")
    training_group.add_argument(
        '--is_noisy_label', type=int, default="0",
        help="If GAN is used, and this is True, the labels will be noisy")
    training_group.add_argument(
        '--use_fk_loss', type=int, default="1",
        help="Use loss in fk space or not, 0 for False and 1 for True")

    # Misc arguments.
    misc_group = parser.add_argument_group('Miscellaneous')

    misc_group.add_argument(
        '--eval_interval', type=int, default=10,
        help="evaluate on test set every eval_interval epochs")
    misc_group.add_argument(
        '--save_interval', type=int, default=10,
        help="Save images every SAVE_INTERVAL epochs")
    misc_group.add_argument(
        '--device', type=str, default="cpu",
        help="Training device 'cpu' or 'cuda:0'")
    misc_group.add_argument(
        '--experiment_num', type = int, default=17,
        help="Id of the experiment running")
    misc_group.add_argument(
        "--is_optimisation", type=int, default=0,
        help="True or False for whether the run is called by the hyperopt"
    )

    args = parser.parse_args()

    main(args)
