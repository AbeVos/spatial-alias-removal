import os
import argparse
import json
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from dataset import Data


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
            args.num_res_blocks, dataset.output_dim,
            args.latent_dim).to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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

    # Misc arguments.
    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument(
        '--eval_interval', type=int, default=2,
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
