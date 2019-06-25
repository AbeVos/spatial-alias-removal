"""
This script trains a model on different amounts of data to analyse the
influence of number of samples on the model's performance.
"""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

from itertools import chain

from train import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model on different amounts of data to analyse"
        " the influence of number of samples on the models' performance.")

    parser.add_argument(
        '-f', dest='argument_file', default='test_arguments.txt',
        help="The argument file contains all arguments needed to train "
        "the model.")
    parser.add_argument(
        '-n', dest='n_experiments', default=20,
        help="Number of experiments.")
    parser.add_argument(
        '--device', default='cuda',
        help="The device to train the models on.")
    args = parser.parse_args()

    with open(args.argument_file) as json_file:
        arguments = json.load(json_file)

    # Make sure the training will return its results.
    arguments["experiment_num"] = 1000
    arguments["is_optimisation"] = 0
    arguments["device"] = args.device
    arguments["eval_interval"] = arguments["n_epochs"] - 1

    psnr_plot = []

    # Create percentages to experiment on.
    percentages = np.linspace(0.1, 0.9, args.n_experiments)
    percentages_display = [f"{100*(1-value):.1f}%" for value in percentages]
    for idx, percentage in enumerate(percentages):
        arguments["test_percentage"] = float(percentage)
        print(f"Training with test percentage {100 * float(percentage):0.1f}%")

        # Create a new parser and fill it with the arguments from the file.
        model_parser = argparse.ArgumentParser()

        for arg, value in arguments.items():
            model_parser.add_argument(
                f'--{arg}', type=type(value), required=False)

        argument_values = zip(
            [f'--{name}' for name in arguments.keys()],
            [str(value) for value in arguments.values()])
        argument_values = list(chain(*argument_values))

        model_args = model_parser.parse_args(argument_values)

        # Train the model with the loaded arguments.
        plot_log, _, _ = main(model_args)
        psnr = plot_log['psnr_val'][-1]

        # psnr = np.random.random(1)
        psnr_plot.append(psnr)

        plt.figure()
        plt.plot(percentages_display[:idx+1], psnr_plot)
        plt.xlabel("Ratio of data used for training")
        plt.ylabel("Test set PSNR")
        plt.savefig("less_data.png")
        plt.close()
