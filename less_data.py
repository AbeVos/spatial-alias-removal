"""
This script trains a model on different amounts of data to analyse the
influence of number of samples on the model's performance.
"""
import argparse
import json
import numpy as np

from itertools import chain

from train import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model on different amounts of data to analyse"
        " the influence of number of samples on the models' performance.")

    parser.add_argument(
        dest='argument_file',
        help="The argument file contains all arguments needed to train "
        "the model.")
    parser.add_argument(
        '-n', dest='n_experiments',
        help="Number of experiments.")
    parser.add_argument(
        '--device', default='cpu',
        help="The device to train the models on.")
    args = parser.parse_args()

    with open(args.argument_file) as json_file:
        arguments = json.load(json_file)

    # Make sure the training will return its results.
    arguments["experiment_num"] = 1000
    arguments["is_optimisation"] = 1
    arguments["device"] = args.device

    for percentage in np.linspace(0, 1, args.n_experiments):
        arguments["test_percentage"] = float(percentage)

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
        print(model_args)


        # Train the model with the loaded arguments.
        plot_log, _, _ = main(model_args)
        psnr = plot_log['psnr_val'][-1]
        print(psnr)
