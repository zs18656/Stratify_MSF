from TS_functions import load_basicts_data
from main_experiment_code import experiment_save_load_models
from forecasting_functions import torch_simple_MLP, torch_simple_RNN, torch_simple_LSTM, torch_simple_Transformer, sklearn_RF, sklearn_XGB
import numpy as np
import argparse
from joblib import cpu_count

n_workers = cpu_count()
print(f"Number of available workers: {n_workers}")

parser = argparse.ArgumentParser(description='Process input parameters for time series forecasting.')

# Add arguments with default values
parser.add_argument('--dataset_name', type=str, default='mg_10000', 
                    help='Name of the dataset (default: mg_10000)')
parser.add_argument('--func', type=str, default='MLP', 
                    help='Function to use (default: MLP)')
parser.add_argument('--metric', type=str, default='mse', 
                    help='Metric to evaluate (default: mse)')
parser.add_argument('--verbose', type=bool, default=True, 
                    help='Verbose mode (default: True)')
parser.add_argument('--H_ahead', type=int, default=10, 
                    help='Horizon for ahead prediction (default: 10)')
parser.add_argument('--window_ratio', type=float, default=2, 
                    help='Window ratio for determining window size (default: 2)')
parser.add_argument('--train_p', type=float, default=0.8, 
                    help='Train proportion (default: 0.8)')
parser.add_argument('--val_p', type=float, default=0.1, 
                    help='Validation proportion (default: 0.1)')
parser.add_argument('--test_p', type=float, default=0.1, 
                    help='Test proportion (default: 0.1)')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs for training (default: 1000)')
parser.add_argument('--window_size', type=int, default=None,
                    help='Window size for sliding window (default: None)')
parser.add_argument('--seed', type = int, default = 0, 
                    help = 'Random seed for reproducibility (default: 0)')
parser.add_argument('--check_code', type = bool, default = False,
                    help = 'toy run with tiny data (default: False)')
parser.add_argument('--recmo_only', type = bool, default = False,
                    help = 'Run only RECMO (default: False)')
 

# Parse arguments
args = parser.parse_args()
print(args)

# Load the univariate time series using the dataset name
dataset_name = args.dataset_name
univariate_time_series = None
try:
    univariate_time_series = np.load(f'univariate_time_series/{dataset_name}.npy')
except FileNotFoundError:
    data = load_basicts_data(dataset_name, idx = 0)
    print(data.shape)
    univariate_time_series = np.mean(data, axis=1)
    print(univariate_time_series.shape)

# Extract other parameters
func = args.func
metric = args.metric
verbose = args.verbose
H_ahead = args.H_ahead
window_ratio = args.window_ratio
train_p = args.train_p
val_p = args.val_p
test_p = args.test_p
seed = args.seed
tiny = args.check_code
recmo_only = args.recmo_only

window_size = args.window_size
# Calculate window size
if window_size is None:
    window_size = int(H_ahead * window_ratio)

# Output summary if verbose is set to True
if verbose:
    print(f"Dataset Name: {dataset_name}")
    print(f"Function: {func}")
    print(f"Metric: {metric}")
    print(f"H_ahead: {H_ahead}")
    print(f"Window Ratio: {window_ratio}")
    print(f"Window Size: {window_size}")
    print(f"Train proportion: {train_p}")
    print(f"Validation proportion: {val_p}")
    print(f"Test proportion: {test_p}")

forecasting_function = None
epochs = args.epochs
if func == 'MLP':
    forecasting_function = torch_simple_MLP(128, epochs=epochs)
if func == 'RNN':
    forecasting_function = torch_simple_RNN(128, epochs=epochs)
if func == 'LSTM':
    forecasting_function = torch_simple_LSTM(128, epochs=epochs)
if func == 'Transformer':
    forecasting_function = torch_simple_Transformer(128, epochs=epochs)
if func == 'RF':
    from forecasting_functions import sklearn_RF
    forecasting_function = sklearn_RF(n_jobs=-1, max_depth=epochs)
if func == 'XGB':
    from forecasting_functions import sklearn_XGB
    forecasting_function = sklearn_XGB(n_jobs=-1)
    
assert forecasting_function is not None, 'Select a forecasting function'

save_folder = 'stratify_results/'
rel_directory = dataset_name + '/' + forecasting_function.name + f'_seed{seed}' + '/'
if tiny:
    rel_directory = f'tiny_{dataset_name}' + '/' + forecasting_function.name + f'_seed{seed}' + '/'

strategy_types = ['RECMO', 'DIRMO', 'DIRREC']

if recmo_only:
    strategy_types = ['RECMO', 'DIRMO']

if univariate_time_series is not None:    
    print(rel_directory)
    print(univariate_time_series.shape)
    experiment_save_load_models(univariate_time_series, forecasting_function, H_ahead, window_size, train_p, val_p, test_p, metric,
                                strategy_types = strategy_types,
                                rel_directory = rel_directory, verbose = verbose, tiny = tiny)
    
else:
    print(f"Dataset {dataset_name} not found.")