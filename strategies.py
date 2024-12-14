from TS_functions import shiftData, moving_window
from sklearn.metrics import mean_squared_error
import numpy as np
from copy import deepcopy
import pandas as pd
from scipy.stats import rankdata
from tqdm import tqdm
import torch
import time
from joblib import dump, load


def mse(preds, ys):
        return (preds - ys)**2

class RECMO():
    
    def __init__(self, function_family, H_ahead, s_parameter):
        assert H_ahead%s_parameter == 0, 'select s such that s divides H_ahead'
        
        self.function_family = function_family
        self.no_recursions = H_ahead//s_parameter
        self.MO_size = s_parameter
        self.H_ahead = H_ahead
        self.model = self.function_family
    
    def fit(self, windowed_data, ys, save_location = ''):
        
        xs, ys = windowed_data, ys
        
        file = f'{save_location}recmo{self.MO_size}'
        try: # first try to load a model if it exists    
            if self.function_family.name in ['MLP', 'RNN', 'LSTM', 'Transformer']:
                dys = ys[:, : self.MO_size]
                self.model.fit(xs, dys, init_only = True) if self.MO_size != 1 else self.model.fit(xs, dys.ravel(), init_only = True)
                
                self.model.load_state_dict(torch.load(f'{file}.pth'))
                print(f'loaded from {file} .pth')
                
            elif self.function_family.name == 'RF':
                self.model = load(file)
                print(f'loaded from {file} .joblib')
                
            elif self.function_family.name == 'XGB':
                self.model.load_model(f'{file}.json')
                print(f'loaded from {file} .json')
                
            else:
                raise ValueError(f'No pretrained model found at {file}')
            
        except:
            dys = ys[:, : self.MO_size]
            self.model.fit(xs, dys) if self.MO_size != 1 else self.model.fit(xs, dys.ravel())
            
            if len(save_location) > 0:
                if self.function_family.name in ['MLP', 'RNN', 'LSTM', 'Transformer']:
                    torch.save(self.model.state_dict(), f'{file}.pth')
                elif self.function_family.name == 'RF':
                    dump(self.model, f'{file}.joblib', compress=('gzip', 3))  # Compression level 3 is a good balance
                elif self.function_family.name == 'XGB':
                    self.model.save_model(f'{file}.json')

    def predict(self, windowed_data):
        if self.no_recursions == 1:
            return self.model.predict(windowed_data)
        
        preds = np.concatenate([windowed_data, np.zeros((windowed_data.shape[0], self.H_ahead))], axis = 1)
        window_size = len(windowed_data[0])
        for recursion_id in range(self.no_recursions):
            input_window = preds[:, recursion_id*self.MO_size: recursion_id*self.MO_size + window_size]
            preds_i = self.model.predict(input_window)
            if self.MO_size == 1:
                preds_i = preds_i.reshape(-1,1)
            preds[:, window_size + recursion_id*self.MO_size: window_size + (recursion_id+1)*self.MO_size] = preds_i
        preds = preds[:, -self.H_ahead:]
        return preds

    def evaluate(self, windowed_data, metric = mean_squared_error):
        xs, ys = shiftData(windowed_data[:-1], self.H_ahead)
        pred_ys = self.predict(xs)
        return np.array([metric(pred_ys[i], ys[i]) for i in range(len(pred_ys))])
    


class DIRMO():
    def __init__(self, function_family, H_ahead, s_parameter):
        assert H_ahead%s_parameter == 0, 'select s such that s divides H_ahead'
        
        self.function_family = function_family
        self.no_funcs = H_ahead//s_parameter
        self.MO_size = s_parameter
        self.H_ahead = H_ahead
        self.models = [deepcopy(self.function_family) for func in range(self.no_funcs)]

    def fit(self, windowed_data, ys, save_location = ''):
        
        xs, ys = windowed_data, ys
            
            
        for func_id, func in enumerate(self.models):
            file = f'{save_location}dirmo{self.MO_size}_id{func_id}'
            try: # first try to load a model if it exists
                if self.function_family.name in ['MLP', 'RNN', 'LSTM', 'Transformer']:
                    dys = ys[:, self.MO_size*(func_id): self.MO_size*(func_id + 1)]
                    func.fit(xs, dys, init_only = True) if self.MO_size != 1 else func.fit(xs, dys.ravel(), init_only = True)
                    
                    func.load_state_dict(torch.load(f'{file}.pth'))
                    print(f'loaded from {file} .pth')
                elif self.function_family.name == 'RF':
                    func = load(f'{file}.joblib')
                    print(f'loaded from {file} .joblib')
                elif self.function_family.name == 'XGB':
                    func.load_model(f'{file}.json')
                    print(f'loaded from {file} .json')
                else:
                    raise ValueError(f'No pretrained model found at {file}')
                
            except:  
                dys = ys[:, self.MO_size*(func_id): self.MO_size*(func_id + 1)]
                func.fit(xs, dys) if self.MO_size != 1 else func.fit(xs, dys.ravel())
                
                if len(save_location) > 0: # save the model
                    if self.function_family.name in ['MLP', 'RNN', 'LSTM', 'Transformer']:
                        torch.save(func.state_dict(), f'{file}.pth')
                    elif self.function_family.name == 'RF':
                        dump(func, f'{file}.joblib', compress=('gzip', 3))
                    elif self.function_family.name == 'XGB':
                        func.save_model(f'{file}.json')
                        
                    

    def predict(self, windowed_data):
        preds = np.zeros([windowed_data.shape[0], self.H_ahead])

        for func_id, func in enumerate(self.models):
            preds_i = func.predict(windowed_data)
            if self.MO_size == 1:
                preds_i = preds_i.reshape(-1,1)
            preds[:, self.MO_size*(func_id): self.MO_size*(func_id + 1)] = preds_i
        
        return preds

    def evaluate(self, windowed_data, metric = mean_squared_error):
        xs, ys = shiftData(windowed_data[:-1], self.H_ahead)
        pred_ys = self.predict(xs)
        return np.array([metric(pred_ys[i], ys[i]) for i in range(len(pred_ys))])

class DIRREC():

    def __init__(self, function_family, H_ahead, s_parameter):
        assert H_ahead%s_parameter == 0, 'select s such that s divides H_ahead'
        
        self.function_family = function_family
        self.no_funcs = H_ahead//s_parameter
        self.MO_size = s_parameter
        self.H_ahead = H_ahead
        self.models = [deepcopy(self.function_family) for func in range(self.no_funcs)]

    def fit(self, windowed_data, ys, save_location = ''):
        
        xs, ys = windowed_data, ys
            

        for func_id, func in enumerate(self.models):
            file = f'{save_location}dirrec{self.MO_size}_id{func_id}'
            try:  # first try to load a model if it exists
                    
                if self.function_family.name in ['MLP', 'RNN', 'LSTM', 'Transformer']:
                    dys = ys[:, self.MO_size*(func_id): self.MO_size*(func_id + 1)]
                    func.fit(xs, dys, init_only = True) if self.MO_size != 1 else func.fit(xs, dys.ravel(), init_only = True)
                    func.load_state_dict(torch.load(f'{file}.pth'))
                    print(f'loaded pretrained {save_location}dirrec{self.MO_size}_id{func_id}.pth')
                    xs = np.random.rand(xs.shape[0], xs.shape[1] + self.MO_size)  # add the MO_size to the input to load the next model
                    
                elif self.function_family.name == 'RF':
                    func = load(f'{file}.joblib')
                    print(f'loaded pretrained {file}.joblib')
                elif self.function_family.name == 'XGB':
                    func.load_model(f'{file}.json')
                    print(f'loaded pretrained {file}.json')
                else:
                    raise ValueError(f'No pretrained model found at {file}')
                
            except:
                dys = ys[:, self.MO_size*(func_id): self.MO_size*(func_id + 1)]
                func.fit(xs, dys) if self.MO_size != 1 else func.fit(xs, dys.ravel())
                func_pred = func.predict(xs) if self.MO_size != 1 else func.predict(xs).reshape(-1,1)
                xs = np.concatenate([xs, func_pred], axis = 1) # add the MO_size to the input to train the next model
            
                if len(save_location) > 0:
                    if self.function_family.name in ['MLP', 'RNN', 'LSTM', 'Transformer']:
                        torch.save(func.state_dict(), f'{file}.pth')
                    elif self.function_family.name == 'RF':
                        dump(func, f'{file}.joblib', compress=('gzip', 3))
                    elif self.function_family.name == 'XGB':
                        func.save_model(f'{file}.json')
                    
                        
    def predict(self, windowed_data):
        preds = np.zeros([windowed_data.shape[0], self.H_ahead])
        for func_id, func in enumerate(self.models):
            preds_i = func.predict(windowed_data) if self.MO_size != 1 else func.predict(windowed_data).reshape(-1,1)
            windowed_data = np.concatenate([windowed_data, preds_i], axis = 1)
            if self.MO_size == 1:
                preds[:, func_id] = preds_i.reshape(-1)
            else: 
                preds[:, self.MO_size*(func_id): self.MO_size*(func_id + 1)] = preds_i
        
        return preds

    def evaluate(self, windowed_data, metric = mean_squared_error):
        xs, ys = shiftData(windowed_data[:-1], self.H_ahead)
        pred_ys = self.predict(xs)
        return np.array([metric(pred_ys[i], ys[i]) for i in range(len(pred_ys))])
        

class STRATIFY():

    def __init__(self, base_forcaster, residual_forecaster):
        self.base_forcaster = base_forcaster
        self.residual_forecaster = residual_forecaster
        
    def fit(self, windowed_data, ys, save_location = ''):
        
        xs, ys = windowed_data, ys
        try:
            base_preds = self.base_forcaster.predict(xs)
        except:
            print('failiure in base, fit base first and use .predict class method') 
            base_preds = self.base_forcaster.predict(xs)
        
        errors = np.subtract(base_preds, ys)
        
        self.residual_forecaster.fit(xs, errors, save_location = save_location)
                        
    def predict(self, windowed_data):
        base_preds = self.base_forcaster.predict(windowed_data) 
        residual_preds = self.residual_forecaster.predict(windowed_data)
        preds = base_preds - residual_preds
        return preds

    def evaluate(self, windowed_data, metric = mean_squared_error):
        xs, ys = shiftData(windowed_data[:-1], self.H_ahead)
        pred_ys = self.predict(xs)
        return np.array([metric(pred_ys[i], ys[i]) for i in range(len(pred_ys))])


class FixedEnsemble():
    def __init__(self, strategy_list):
        self.strategy_list = strategy_list
        self.weights = np.ones(len(strategy_list))/len(strategy_list)
        
    def fit(self, xs, ys):
        preds = np.array([self.strategy_list[strat_id].predict(xs) for strat_id in range(len(self.strategy_list))])
        preds_by_strat = preds.reshape(preds.shape[0], -1)
        if ys is not None:
            ys = ys.reshape(-1)
            weights = np.linalg.lstsq(preds_by_strat.T, ys, rcond=None)[0]
            self.weights = weights
        
    def predict(self, xs):
        preds = np.array([self.strategy_list[strat_id].predict(xs) for strat_id in range(len(self.strategy_list))])
        return np.array([weight * preds[idx] for idx, weight in enumerate(self.weights)]).sum(axis = 0)