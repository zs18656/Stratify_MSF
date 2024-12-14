from strategies import RECMO, DIRMO, DIRREC, STRATIFY, FixedEnsemble
from TS_functions import getMackey, create_sliding_windows, factors
from copy import deepcopy
import pandas as pd
import numpy as np
from tqdm import tqdm
import os   

def mse(preds, ys):
        return (preds - ys)**2
    
def get_stats(preds):
        average_over_forecast = preds.mean(axis = 2).T
        mean_over_strategies = average_over_forecast.mean(axis = 0)
        median_over_strategies = np.median(average_over_forecast, axis = 0)
        upper_quartile = np.percentile(average_over_forecast, 75, axis = 0)
        lower_quartile = np.percentile(average_over_forecast, 25, axis = 0)
        std_over_strategies = np.std(average_over_forecast, axis = 0)
        return mean_over_strategies, median_over_strategies, upper_quartile, lower_quartile, std_over_strategies
         
def experiment_save_load_models(univariate_time_series, forecasting_function, H_ahead, window_size, train_p, val_p, test_p, metric, 
                                strategy_types,
                                rel_directory = None, verbose = True, tiny = False):

    input_windows, output_windows = create_sliding_windows(univariate_time_series, window_size, H_ahead)
    train_N, val_N, test_N = int(train_p*len(input_windows)), int(val_p*len(input_windows)), int(test_p*len(input_windows))
    forc_xs, forc_ys = input_windows[:train_N], output_windows[:train_N]
    val_xs, val_ys = input_windows[train_N:train_N+val_N], output_windows[train_N:train_N+val_N]
    test_xs, test_ys = input_windows[train_N+val_N:train_N+val_N+test_N], output_windows[train_N+val_N:train_N+val_N+test_N]
    
    if tiny:
        forc_xs, forc_ys = forc_xs[:100], forc_ys[:100]
        val_xs, val_ys = val_xs[:100], val_ys[:100]
        test_xs, test_ys = test_xs[:100], test_ys[:100]

    if verbose:
        print(f"Train data shape: {forc_xs.shape, forc_ys.shape}")
        print(f"Validation data shape: {val_xs.shape, val_ys.shape}")
        print(f"Test data shape: {test_xs.shape, test_ys.shape}")

    save_folder = 'torch_models/'
    directory = save_folder + rel_directory

    try:
        os.makedirs(directory, exist_ok=True)
        print(f"The directory '{directory}' is ready (either it existed or was created).")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    strategy_list = []
    strategy_names = []
    strategy_types = strategy_types

    for s in factors(H_ahead)[:-1]:
        if 'RECMO' in strategy_types:
            strategy_list.append(RECMO(deepcopy(forecasting_function), H_ahead, s))
            strategy_names.append(f"RECMO_{s}")
        if 'DIRMO' in strategy_types:
            strategy_list.append(DIRMO(deepcopy(forecasting_function), H_ahead, s))
            strategy_names.append(f"DIRMO_{s}")
        if 'DIRREC' in strategy_types:
            strategy_list.append(DIRREC(deepcopy(forecasting_function), H_ahead, s))
            strategy_names.append(f"DIRREC_{s}")
    
    s = H_ahead
    strategy_list.append(RECMO(deepcopy(forecasting_function), H_ahead, s))
    strategy_names.append(f"RECMO_{s}")  
    
    n_base_strategies = len(strategy_list)
    
    print('fitting base strategies')
    [x.fit(forc_xs, forc_ys, save_location = directory) for x in tqdm(strategy_list[:n_base_strategies])]
    
    n_single_strategies = len(strategy_list)
    for i in range(n_single_strategies):
        for j in range(n_single_strategies):
            pair = (strategy_names[i], strategy_names[j])
            # for stratify, we can only copy the base model. deep copy for residual forecaster because we dont want to reassign weights of base models
            strategy_list.append(STRATIFY(strategy_list[i], deepcopy(strategy_list[j])))  
            strategy_names.append(f"stratify{pair[0]}_{pair[1]}")
            
    assert len(strategy_names) == len(strategy_list)

    if verbose:
        print(strategy_names)

    [x.fit(forc_xs, forc_ys, save_location = directory + strategy_names[n_base_strategies+idx]) for idx, x in tqdm(enumerate(strategy_list[n_base_strategies:]))]
    
    average_ensemble = FixedEnsemble(strategy_list)
        
    fixed_ensembles = [average_ensemble]
    fixed_ensemble_names = ['average_ensemble']
    
    strategy_list_all = strategy_list + fixed_ensembles
    strategy_names_all = strategy_names + fixed_ensemble_names
    
    print('train preds')
    train_preds = [x.predict(forc_xs) for x in tqdm(strategy_list_all)]
    val_preds = [x.predict(val_xs) for x in tqdm(strategy_list_all)]
    print('val preds')
    train_errors = np.array([mse(preds, forc_ys) for preds in train_preds])
    val_errors = np.array([mse(preds, val_ys) for preds in val_preds])
    print('getting stats')
    train_mean_over_strategies, train_median_over_strategies, train_upper_quartile, train_lower_quartile, train_std_over_strategies = get_stats(train_errors)
    val_mean_over_strategies, val_median_over_strategies, val_upper_quartile, val_lower_quartile, val_std_over_strategies = get_stats(val_errors)
    stats_df_train = pd.DataFrame([train_mean_over_strategies, train_median_over_strategies, train_upper_quartile, train_lower_quartile, train_std_over_strategies], columns=strategy_names_all, index=['Mean', 'Median', 'Upper quartile', 'Lower quartile', 'Standard deviation'])
    stats_df_val = pd.DataFrame([val_mean_over_strategies, val_median_over_strategies, val_upper_quartile, val_lower_quartile, val_std_over_strategies], columns=strategy_names_all, index=['Mean', 'Median', 'Upper quartile', 'Lower quartile', 'Standard deviation'])
    print('saving')
    if verbose:
        print(stats_df_train.head())
        print(stats_df_val.head())
        
    save_folder = 'stratify_results/'
    directory = save_folder + rel_directory    
    try:
        os.makedirs(directory, exist_ok=True)
        print(f"The directory '{directory}' is ready (either it existed or was created).")
    except Exception as e:
        print(f"An error occurred: {e}")
    save_string = directory + f'{metric}_{H_ahead}_{window_size}_{train_p}'
    train_string = save_string + '_train.csv'
    val_string = save_string + '_val.csv'
    stats_df_train.to_csv(train_string)
    stats_df_val.to_csv(val_string)
