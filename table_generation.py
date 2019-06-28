from train import save_loss_plot, plot_samples
from train import main as main_model
import argparse
import os
import pickle
import itertools
import numpy as np

def get_args_from_params(params):
    """
    Transforms parameters from grid search into arguments used by main.

    Parameters
    ----------
    params : list of len 3, with categorical values for following arguments:
    criterion type : 0 for MSE , 1 for L1 and 2 for None
    is_gan: 0 and 1
    fk_loss: 0 and 1
    """
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
        '--latent_dim', type=int, default=128,
        help="dimensionality of the latent space, only relevant for "
             "EDSR and VDSR")
    model_group.add_argument(
        '--num_res_blocks', type=int, default=4,
        help="Number of resblocks in model, only relevant for EDSR and VDSR")

    # Training arguments.
    training_group = parser.add_argument_group('Training')

    training_group.add_argument(
        '--n_epochs', type=int, default=80,
        help="number of epochs")
    training_group.add_argument(
        '--batch_size', type=int, default=8,
        help="batch size")
    training_group.add_argument(
        '--lr', type=float, default=0.001,
        help="learning rate")
    training_group.add_argument(
        '--scheduler_patience', type=int, default="5",
        help="How many val epochs of no improvement to consider Plateau")
    training_group.add_argument(
        '--is_psnr_step', type=int, default="1",
        help="Use PSNR for scheduler or separate losses")


    criterion_dict ={0: "MSE", 1: "L1", 2: "None"}
    criterion = criterion_dict[params[0]]
    training_group.add_argument(
        '--criterion_type', type=str, default=f"{criterion}",
        choices=['MSE', 'L1', 'None'],
        help="Reconstruction criterion to use.")

    training_group.add_argument(
        '--is_gan',  type=int, default=f"{params[1]}",
        help="If set, use GAN loss.")
    training_group.add_argument(
        '--is_noisy_label', type=int, default="0",
        help="If GAN is used, and this is True, the labels will be noisy")
    training_group.add_argument(
        '--use_fk_loss', type=int, default=f"{params[2]}",
        help="Use loss in fk space or not, 0 for False and 1 for True")

    # Misc arguments.
    misc_group = parser.add_argument_group('Miscellaneous')

    misc_group.add_argument(
        '--eval_interval', type=int, default=4,
        help="evaluate on test set every eval_interval epochs")
    misc_group.add_argument(
        '--save_interval', type=int, default=10,
        help="Save images every SAVE_INTERVAL epochs")
    misc_group.add_argument(
        '--device', type=str, default="cuda:0",
        help="Training device 'cpu' or 'cuda:0'")
    misc_group.add_argument(
        '--experiment_num', type=int, default=29,
        help="Id of the experiment running")
    misc_group.add_argument(
        "--is_optimisation", type=int, default=1,
        help="True or False for whether the run is called by the hyperopt"
    )
    misc_group.add_argument(
        "--save_test_dataset", type=int, default=0,
        help="True or False for option to save test dataset "
    )

    args = parser.parse_args()
    return args


def main(args):
    lst = np.array(list(itertools.product([0, 1, 2], repeat=3)))
    params = np.concatenate((lst[:7], lst[9:15]))[:, [2,0,1]]
    results = []
    for param in params:
        if param[0]==2 and param[1]==0:

            print(f"Params: {param}, No reconstruction loss and no GAN")
        else:
            args = get_args_from_params(param)
            plot_log, __, __ = main_model(args)
            print(f"Parameters {param} got PSNR of {plot_log['psnr_val'][-1]} and SSIM of {plot_log['ssim_val'][-1]}")
            results.append((plot_log['psnr_val'][-1], plot_log['ssim_val'][-1]))

    with open('table.txt', 'w') as f:
        for item in results:
            f.write(f"{item}\n" )
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    args = parser.parse_args()
    main(args)