
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import torch
from train import save_loss_plot, plot_samples
from train import main as main_model
import argparse
import os
import pickle

# function to make turn params sample into arguments
def get_args_from_params(params):
    """
    Transforms parameters from optimsation into arguments used by main.

    Parameters
    ----------
    params : dict
        Sampled parameters.
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
        '--model', type=str, default=f"{params['model_choice']['model']}",
        choices=['EDSR', 'SRCNN', "VDSR"],
        help="Model type.")
    model_group.add_argument(
        '--latent_dim', type=int, default=int(params['model_choice']['latent_dim']),
        help="dimensionality of the latent space, only relevant for EDSR and VDSR")
    model_group.add_argument(
        '--num_res_blocks', type=int, default=int(params['model_choice']['num_res_blocks']),
        help="Number of resblocks in model, only relevant for EDSR and VDSR")



    # Training arguments.
    training_group = parser.add_argument_group('Training')

    training_group.add_argument(
        '--n_epochs', type=int, default=int(params['num_epochs']),
        help="number of epochs")
    training_group.add_argument(
        '--batch_size', type=int, default=int(params['batch_size']),
        help="batch size")
    training_group.add_argument(
        '--lr', type=float, default=params['lr'],
        help="learning rate")
    training_group.add_argument(
        '--scheduler_patience', type=int, default=params['scheduler_patience'],
        help="How many epochs of no improvement to consider Plateau")
    training_group.add_argument(
        '--is_psnr_step', type=int, default=int(params['is_psnr_step']),
        help="Use PSNR for scheduler or separate losses")
    training_group.add_argument(
        '--criterion_type', type=str, default=params['criterion_type'],
        choices=['MSE', 'L1', 'None'],
        help="Reconstruction criterion to use.")
    training_group.add_argument(
        '--is_gan',  type = int, default=int(params['is_gan']),
        help="If set, use GAN loss.")
    training_group.add_argument(
        '--is_noisy_label', type=int, default=int(params['is_noisy_label']),
        help="If GAN is used, and this is True, the labels will be noisy")
    training_group.add_argument(
        '--use_fk_loss', type=int, default=int(params['is_fk_loss']),
        help="Use loss in fk space or not, 0 for False and 1 for True")

    # Misc arguments.
    misc_group = parser.add_argument_group('Miscellaneous')

    misc_group.add_argument(
        '--eval_interval', type=int, default=int(params['eval_interval']),
        help="evaluate on test set every eval_interval epochs")
    misc_group.add_argument(
        '--save_interval', type=int, default=10,
        help="Save images every SAVE_INTERVAL epochs")
    misc_group.add_argument(
        '--device', type=str, default="cuda:0",
        help="Training device 'cpu' or 'cuda:0'")
    misc_group.add_argument(
        '--experiment_num', type=int, default=0,
        help="Id of the experiment running")
    misc_group.add_argument(
        "--is_optimisation", type=int, default=1,
        help="True or False for whether the run is called by the hyperopt"
    )

    args = parser.parse_args()

    return args



def main(hyper_args):
    #objective is a surrogate function, to be optimised
    def objective(params):

        #if impossible combination, return a fail
        if params['criterion_type'] =='None' and params['is_gan']==0:
            print("Unfortunately, this combination of parameters is not possible.")
            return {'loss': 0, 'status': 'fail'}
        #getting params
        iter_num = len(trials.trials)
        print(f"Starting {iter_num} a run with parameters: \n {params}")
        args = get_args_from_params(params)
        #actually training and evaluating the model
        plot_log, generator, test_data = main_model(args)
        #getting proxy loss, PSNR, negative, because we're minimising.
        loss = -plot_log["psnr_val"][-1]
        print(f"Iteration {iter_num} got psnr of {-loss}")
        #saving images for this iteration
        plot_samples(generator, test_data, iter_num,  torch.device("cuda:0"),
                     f"{results_directory}/images")

        save_loss_plot(plot_log['G'], plot_log['D'], f"{results_directory}/losses", name=iter_num)
        save_loss_plot(plot_log['G_val'], plot_log['D_val'], f"{results_directory}/losses", is_val=True, name=iter_num)
        #saving models actual losses
        torch.save(generator, os.path.join(f"{results_directory}/models", f'generator_{iter_num}.pth'))
        #saving trials, here, because can be the case that stops inbetween evaluations
        pickle.dump(trials, open(f"{results_directory}/{hyper_args.trails_name_to_save}.pkl", "wb"))
        return {
            'loss': loss,
            'status': STATUS_OK, #'ok' or 'fail'
        }

    #initial distribution of hyper-parameter space
    space ={
        "num_epochs":hp.quniform('num_epochs', 1, 40, 1),
        "model_choice": hp.choice(
            'model_choice', [{
                'model': 'EDSR',
                'latent_dim': hp.quniform('latent_dim_edsr', 32, 257, 1),
                'num_res_blocks': hp.quniform('num_res_block_edsr', 1, 12, 1)
            }, {
                'model': 'VDSR',
                'latent_dim': hp.quniform('latent_dim_vdsr', 32, 257, 1),
                'num_res_blocks': hp.quniform('num_blocks_vdsr', 1, 12, 1)
            }, {
                'model': 'SRCNN',
                'latent_dim': 0,
                'num_res_blocks': 0
            }]),
        "batch_size": hp.quniform("batch_size", 2, 12, 1),
        "lr": hp.loguniform("lr", -15, 0),
        "scheduler_patience": hp.quniform("scheduler_patience", 1, 30, 1),
        "is_psnr_step": hp.randint("is_psnr_step", 2),
        "criterion_type": hp.choice("criterion_type", ["MSE", "L1", "None"]),
        "is_gan": hp.randint("is_gan", 2),
        "is_noisy_label": hp.randint("is_noisy_label", 2),
        "is_fk_loss": hp.randint("is_fk_loss", 2),
        "eval_interval": hp.quniform("eval_interval", 1, 15, 1)
    }

    #getting directory
    results_directory = hyper_args.results_folder
    os.makedirs(results_directory, exist_ok=True)
    os.makedirs(f"{results_directory}/losses", exist_ok=True)
    os.makedirs(f"{results_directory}/images", exist_ok=True)
    os.makedirs(f"{results_directory}/models", exist_ok=True)

    #initialise trials
    if hyper_args.previous_trials_name is None:
        trials = Trials()
        max_evals =hyper_args.eval_number
        print("Initialising new trials")
    else:
        #load trials
        trials = pickle.load(open(f"{results_directory}/{hyper_args.previous_trials_name}.pkl", "rb"))
        max_evals =hyper_args.eval_number + len(trials)


    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials)
    pickle.dump(trials, open(f"{results_directory}/{hyper_args.trails_name_to_save}.pkl", "wb"))
    print(best)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--eval_number", type=int, default=10,
                        help="Number of iterations the hyperoptimisation will run for")
    parser.add_argument("--results_folder", type = str, default="hyper_results",
                        help="Folder where all the results of the optimisation will reside")
    parser.add_argument("--previous_trials_name", type = str, default="trials",
                    help="Name of trials to start with. If none, starts from scratch")
    parser.add_argument("--trails_name_to_save", type = str, default="trials",
                        help="Name of trials to save the results to.")


    hyper_args = parser.parse_args()
    main(hyper_args)
