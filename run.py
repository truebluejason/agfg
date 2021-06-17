import argparse
import numpy as np
import os

from autogluon_benchmark.tasks import task_loader, task_utils
from autogluon.features.generators import PipelineFeatureGenerator
from generators.autoencoder import AutoEncoderFeatureGenerator,\
    DenoisingAutoEncoderFeatureGenerator

"""
Test AutoGluon Feature Generator
Example Command: # python3 run.py -m default -t robert -n base300
"""

OPENML_TASK_NAME_TO_ID = {
    'robert': 168332,
    'riccardo': 168338,
    'guillermo': 168337,
    'dilbert': 168909
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # General Configs
    parser.add_argument('-n', '--name', help='run name',
                        type=str, required=True)
    parser.add_argument('-m', '--mode', help='which feature generation to use',
                        type=str, choices=['default', 'ae', 'dae'],
                        required=True)
    parser.add_argument('-t', '--task', help='openml task name',
                        type=str, choices=['robert'], required=True)
    parser.add_argument('-f', '--fold', help='number of train/test splits',
                        type=int, default=1)
    # Autoencoder Configs
    parser.add_argument('-q', '--epochs', help='number of training epochs',
                        type=int, default=10)
    parser.add_argument('-e', '--e_dim', help='embedding dimension',
                        type=int, default=256)
    parser.add_argument('-r', '--runtime', help='runtime min (None=no limit)',
                        type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('scores'):
        os.makedirs('scores')

    # Setup arguments
    task_dict = task_loader.get_task_dict()
    task_name = args.task  # task name in yaml @ autogluon_benchmark/tasks
    task_id = OPENML_TASK_NAME_TO_ID[task_name]
    n_folds = args.fold
    model_path = os.path.join('models', args.name)
    score_path = os.path.join('scores', args.task)

    init_args = {'path': model_path}
    fit_args = {
        'verbosity': 2,
        'presets': ['medium_quality_faster_train'],  # ['best_quality']
        'excluded_model_types': ['FASTAI', 'XGB', 'GBM']
    }

    if args.runtime is not None:
        fit_args['time_limit'] = args.runtime * 60

    if args.mode == 'ae':
        custom_generator = AutoEncoderFeatureGenerator(
            e_dim=args.e_dim, n_epochs=args.epochs)
        fit_args['feature_generator'] = PipelineFeatureGenerator(
            generators=[[custom_generator]])
    elif args.mode == 'dae':
        custom_generator = DenoisingAutoEncoderFeatureGenerator(
            noise_stdev=0.1, e_dim=args.e_dim, n_epochs=args.epochs)
        fit_args['feature_generator'] = PipelineFeatureGenerator(
            generators=[[custom_generator]])

    # Download and run AutoGluon on OpenML
    predictors, scores = task_utils.run_task(
        task_id, n_folds=args.fold, init_args=init_args, fit_args=fit_args)

    # Compute score and save
    score = float(np.mean(scores))
    if len(scores) > 1:
        score_std = np.std(scores, ddof=1)
    else:
        score_std = 0.0  # Should this be np.inf?
    msg = f'{args.mode} score ({args.fold} folds): {round(score, 5)}' + \
        f'(+- {round(score_std, 5)})\n'
    print(msg)

    for i, predictor in enumerate(predictors):
        leaderboard = predictor.leaderboard()
        leaderboard.to_csv(os.path.join(
            model_path, f'leaderboard_{i}.csv'), index=False)

    with open(score_path, "a") as fp:
        fp.write(msg)
