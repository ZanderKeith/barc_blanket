"""Perform a grid search


"""

"""Run a parameter sweep using Optuna.

A sweep is defined in a 'sweep_config.yaml' file with the following structure:
    ```
    metric: metric name
    direction: minimize or maximize
    parameters:
        # https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html
        parameter1:
        distribution: int
        max: 100
        min: 0
        parameter2:
        distribution: float
        log: true for log scale, false for linear scale
        max: 1.0
        min: 0.0
        parameter3:
        distribution: categorical
        values:
        - value1
        - value2
    ```

"""

import yaml
import optuna
import argparse
import numpy as np


from barc_blanket.optimize_model import evaluate_metric
from barc_blanket.utilities import working_directory

def _parse_args():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Run a parameter grid search using Optuna")
    parser.add_argument("sweep_directory", type=str, help="Relative path to directory where all the sweep input and output files are stored.")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = _parse_args()

    sweep_directory = args.sweep_directory

    # Change to the sweep directory
    with working_directory(sweep_directory):
    
        # Load the config
        sweep_config = yaml.safe_load(open(f"sweep_config.yaml", "r"))

        # Define the grid of the sweep
        grid = {}
        for parameter, parameter_space in sweep_config['parameters'].items():
            distribution = parameter_space['distribution']
            if distribution in ['int', 'float']:
                grid[parameter] = np.arange(parameter_space['min'], parameter_space['max']+parameter_space['step'], parameter_space['step'])
            elif distribution == 'categorical':
                grid[parameter] = parameter_space['values']
            else:
                raise ValueError(f"Invalid distribution type: {parameter_space['distribution']}")
            
        # Create storage for trial results that can support concurrent writes
        sweep_results_path = f"sweep_results.db"
        lock_obj = optuna.storages.JournalFileOpenLock(sweep_results_path)
        storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(sweep_results_path, lock_obj=lock_obj)
        )

        study = optuna.create_study(
            storage=storage, 
            study_name=f"{sweep_directory}",
            load_if_exists=True,
            sampler=optuna.samplers.GridSampler(grid)
        )

        # Run the grid search until completion
        # This has no limits on the number of trials or time
        study.optimize(lambda trial: evaluate_metric(trial, sweep_config))

if __name__ == "__main__":
    main()