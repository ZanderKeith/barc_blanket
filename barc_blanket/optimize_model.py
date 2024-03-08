import openmc
import optuna

from barc_blanket.models.simple_geometry import make_model

def evaluate_metric(trial, sweep_config):
    """ Objective function for the optimization

    Parameters:
    ----------
    trial : optuna.Trial
        An optuna trial object
    sweep_config : dict
        A dictionary containing the sweep configuration

    Returns:
    -------
    metric_val: float
        The value of the metric calculated for the parameters in this trial
    """

    # Obtain the values of parameters from the trial
    model_config = {}
    parameters = sweep_config['parameters']
    parameter_names = list(parameters.keys())

    for parameter_name in parameter_names:
        parameter = parameters[parameter_name]
        distribution_type = parameter['distribution']

        if distribution_type == "int":
            min = parameter['min']
            max = parameter['max']
            chosen_value = trial.suggest_int(parameter_name, min, max)
        elif distribution_type == "float":
            min = parameter['min']
            max = parameter['max']
            log = parameter['log']
            chosen_value = trial.suggest_float(parameter_name, min, max, log=log)
        elif distribution_type == "categorical":
            values = parameter['values']
            chosen_value = trial.suggest_categorical(parameter_name, values)
        else:
            raise ValueError(f"Invalid distribution type: {distribution_type}")
        
        model_config[parameter_name] = chosen_value

    # Create the model and evaluate the metric
    try:
        model = make_model(model_config)
        metric = sweep_config['metric']
        if metric == "tbr":
            metric_val = tritium_breeding_ratio(model, model_config)
        #elif metric == "some_other_arbitrary_metric":
        #    metric_val = whatever_function(model)
        else:
            raise ValueError(f"Invalid metric: {metric}")
    except MemoryError as e:
        print(f"Ran out of memory for trial {trial.number}")
        print(e)
        metric_val = float('nan')
    except Exception as e:
        print(f"Error in trial {trial.number}, pruning...")
        print(e)
        # If anything oges wrong during training or validation, say that the trial was pruned
        # This should make Optuna try a different set of parameters to avoid errors
        raise optuna.TrialPruned()
    
    return metric_val

# THIS IS JUST A PROOF OF CONCEPT
# THIS FUNCTION DOES NOT PRODUCE CORRECT OUTPUT
def tritium_breeding_ratio(model:openmc.Model, model_config:dict):
    """ THIS IS JUST A PROOF OF CONCEPT
    THIS FUNCTION DOES NOT PRODUCE CORRECT OUTPUT
    
    Calculate the tritium breeding ratio for the given model

    Parameters:
    ----------
    model : openmc.Model
        The model to evaluate the metric for

    model_config : dict
        A dictionary containing the model configuration

    Returns:
    -------
    tbr: float
        The tritium breeding ratio for the model
    """

    # Run the model
    # TODO: stop hardcoding statepoint
    model.run()
    # Get the last statepoint number from batches
    statepoint_number = model_config['batches']
    final_statepoint = openmc.StatePoint(f"statepoint.{statepoint_number}.h5")

    # Get tally results
    # TODO: do this programmatically instead of hardcoded here
    tallies = final_statepoint.tallies
    tbr_tally_id = 2
    tally_result = tallies[tbr_tally_id].mean[0][0][0]
    
    # TODO do some volume weighting or whatever to get an actual TBR
    return tally_result