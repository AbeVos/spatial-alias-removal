import torch
from torch import nn
from math import log10
from scipy.stats import  wilcoxon
import argparse
import os
from models import SRCNN, VDSR, EDSR
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from train import transform_fk
import numpy as np
def add_subplot(plt, image, i,  title=None, cmap='viridis'):
    plt.subplot(1, 3, i)
    plt.title(title)
    plt.imshow(image.squeeze().detach().cpu(),
               interpolation='none', cmap=cmap)
    plt.axis('off')
def main(args):
    is_image = False
    is_diff = True
    #setting the results folder
    results_directory = args.results_folder
    os.makedirs(results_directory, exist_ok=True)
    #loading the model
    device = torch.device(args.device)
    #we first see if there is a model with needed name in our models folder,
    try:
        model = torch.load(f"{args.model_folder}/{args.model_name}", map_location = device)
    #if not, then we load it from the results folder
    except:
        model = torch.load(f"results/result_{args.experiment_num}/{args.model_name}", map_location = device)
        model_baseline = torch.load(f"results/result_{args.experiment_num_baseline}/{args.model_name}", map_location = device)
    model.eval()
    #loading the data
    data_folder_for_results = args.folder_with_data
    tensor_x = torch.load( f'{data_folder_for_results}/data_x_{args.experiment_num}.pt')
    tensor_y = torch.load( f'{data_folder_for_results}/data_y_{args.experiment_num}.pt')
    dataset = TensorDataset(tensor_x, tensor_y)

    def psnr(sr, hr):
        return 10 * log10(1 / nn.functional.mse_loss(sr, hr).item())

    psnr_sures = []
    psnr_bicubic = []
    psnr_baseline = []
    for i, (x,y) in enumerate(dataset):
        lores = x.to(device).float()
        hires = y.to(device).float()
        sures = model(lores.unsqueeze(0)).squeeze(0)
        sures_baseline = model_baseline(lores.unsqueeze(0)).squeeze(0)
        output_dim = hires.shape[1:]

        lores = torch.nn.functional.interpolate(lores.unsqueeze(0), size=output_dim, mode='bilinear')
        if is_diff:
            diff_lor = hires - lores
            diff_sure = hires - sures

            plt.figure(figsize=(10, 20))
            plt.subplot(2, 1, 1)

            plt.imshow(diff_lor.squeeze().detach().cpu(),
               interpolation='none', cmap='gray')
            plt.axis('off')
            plt.subplot(2, 1, 2)

            plt.imshow(diff_sure.squeeze().detach().cpu(),
                   interpolation='none', cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(args.results_folder, f'diff{i}.pdf'))
            plt.close('all')

        elif not is_image:
            psnr_bicubic.append(psnr(lores, hires))
            psnr_sures.append(psnr(sures, hires))
            psnr_baseline.append(psnr(sures_baseline, hires))

        else:
            output_dim = hires.shape[1:]
            #print(output_dim,lores.unsqueeze(0).shape )

            lores = torch.nn.functional.interpolate(lores.unsqueeze(0), size=output_dim).squeeze(0)


            plt.figure(figsize=(30, 10))
            # Plot images.
            add_subplot(plt, lores, 1, "LR", cmap='gray')
            add_subplot(plt, sures, 2, "SR", cmap='gray')
            add_subplot(plt, hires, 3, "HR", cmap='gray')
            plt.tight_layout()
            plt.savefig(os.path.join(args.results_folder, f'results_parabola_{i}.pdf'))
            plt.savefig(os.path.join(args.results_folder, f'results_parabola_{i}.png'))
            plt.close('all')
            # Plot transformed images.
            plt.figure(figsize=(30, 5))
            plt.figure()
            add_subplot(plt, transform_fk(lores, output_dim), 1, "LR fk")
            add_subplot(plt, transform_fk(sures, output_dim), 2,  "SR fk")
            add_subplot(plt, transform_fk(hires, output_dim), 3, "HR fk")
            plt.tight_layout()
            plt.savefig(os.path.join(args.results_folder, f'results_fk_{i}.pdf'))
            plt.savefig(os.path.join(args.results_folder, f'results_fk_{i}.png'))
            plt.close('all')
    result_bicubic = wilcoxon(psnr_sures, psnr_bicubic )
    results_baseline = wilcoxon(psnr_sures, psnr_baseline)
    print(f'Bicubic got average PSNR of {np.mean(psnr_bicubic)}, \n \
          Baseline got average PSNR of {np.mean(psnr_baseline)} \n \
           Model got average PSNR of {np.mean(psnr_sures)} \n \
          Test of bicubic got results {result_bicubic } \n \
          Test of baseline got results {results_baseline } \n \
          ')
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--folder_with_data", type=str, default="final/data",
                        help="Folder with images to be visualised")
    parser.add_argument("--data_name", type=str, default="data.pt",
                        help="Folder with images to be visualised")
    parser.add_argument("--model_folder", type = str, default="final/model",
                        help="Folder with the model to generate SR from")
    parser.add_argument("--model_name", type = str, default="generator.pth",
                        help="The name of the model to generate the images")
    parser.add_argument("--model_name_baseline", type = str, default="generator.pth",
                        help="The name of the model to generate the images")
    parser.add_argument("--results_folder", type = str, default="final/results_27",
                        help="Folder where all the results will be stored")

    parser.add_argument('--device', type=str, default="cuda:0",
                         help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--experiment_num', type=int, default=27,
                         help="Id of the experiment ")
    parser.add_argument('--experiment_num_baseline', type=int, default=29,
                        help="Id of the experiment ")


    args = parser.parse_args()
    main(args)