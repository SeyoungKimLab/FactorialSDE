import sys
sys.path.append('../../')

import os
import os.path as osp
import argparse
from dateutil.relativedelta import relativedelta
from collections import defaultdict
import pickle
from tqdm import tqdm
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

import gpflow
from gpflow.models import SVGP
from gpflow.kernels import (
    SquaredExponential, 
    LinearCoregionalization, 
    SharedIndependent, 
    Matern32,
    Matern52
)
from gpflow.config import default_float, default_jitter
from gpflow.inducing_variables import (
    InducingPoints,
    SharedIndependentInducingVariables,
    SeparateIndependentInducingVariables
)
from gpflow.likelihoods import Gaussian
from gpflow.utilities import (
    print_summary, 
    set_trainable, 
    to_default_float, 
    triangular, 
    positive
)
from gpflow.base import Parameter

from sklearn.model_selection import train_test_split

import fsde
from fsde.baselines import *

# CONSTANTS

RESULTS_DIR = './sim_results'
DATA_PATH = '../../data/sim_small_data.pkl'
IND_PATH = './ind_times.pkl'
N_STEPS = 50000
N_FOLDS = 5

def unpack_data(data):
    train_times = data['train']['times']
    Y_train = data['train']['Y']
    train_idx = data['train']['idx']

    int_times = data['test_int']['times']
    Y_int = data['test_int']['Y']
    int_idx = data['test_int']['idx']

    ext_times = data['test_ext']['times']
    Y_ext = data['test_ext']['Y']
    ext_idx = data['test_ext']['idx']

    train_dataset = (train_times, Y_train)
    int_dataset = (int_times, Y_int)
    ext_dataset = (ext_times, Y_ext)
    datasets = dict(train=train_dataset, int=int_dataset, ext=ext_dataset)
    idxs = dict(train=train_idx, int=int_idx, ext=ext_idx)

    return datasets, idxs

def mean_std(metrics):
    n_runs = len(metrics)
    train_metrics = np.array([metric['train'] for metric in metrics])
    int_metrics = np.array([metric['int'] for metric in metrics])
    ext_metrics = np.array([metric['ext'] for metric in metrics])

    train_avg = np.mean(train_metrics)
    train_std = np.std(train_metrics)
    int_avg = np.mean(int_metrics)
    int_std = np.std(int_metrics)
    ext_avg = np.mean(ext_metrics)
    ext_std = np.std(ext_metrics)

    avg = dict(train=train_avg, int=int_avg, ext=ext_avg)
    std = dict(train=train_std, int=int_std, ext=ext_std)

    return avg, std

def save_results(model_type, results, use_natgrad):
    if model_type == 'lmc_gpr':
        pkl_path = 'lmc_gpr_results_CV.pkl'

    elif model_type == 'lmc_svgp':
        pkl_path = f'lmc_svgp_results_CV_natgrad.pkl' if use_natgrad else f'lmc_svgp_results_CV_adam.pkl'

    elif model_type == 'imc_svgp':
        pkl_path = f'imc_svgp_results_CV_natgrad.pkl' if use_natgrad else f'lmc_svgp_results_CV_adam.pkl'

    pkl_path = osp.join(RESULTS_DIR, pkl_path)
    with open(pkl_path, 'wb') as fh:
        pickle.dump(results, fh)

    print(f'[Small Simulation] {model_type} results saved: {pkl_path}.\n')

def train_LMC_GPR(data, split_id, use_natgrad=None):
    split = data[f'split_{split_id}']
    P = split['train']['Y'].shape[0]
    L = 5
    kernel = 'Matern52'
    
    train_params = dict(kernel_type=kernel,
                        lengthscale=1.,
                        kernel_var=1.,
                        L=5,
                        n_steps=N_STEPS)

    model, opt_log = train_lmc_gpr(train_params, split['train']['times'], split['train']['Y'])

    # Generate predictions on data
    train_preds = lmc_gpr_predict(model, (split['train']['times'], split['train']['Y']))
    int_preds = lmc_gpr_predict(model, (split['test_int']['times'], split['test_int']['Y']))
    ext_preds = lmc_gpr_predict(model, (data['test_ext']['times'], data['test_ext']['Y']))

    preds = dict(train=train_preds, int=int_preds, ext=ext_preds)

    # Compute accuracy and uncertainty metrics
    train_metrics = compute_metrics_lmc_gpr(model, 
                                            (split['train']['times'], split['train']['Y']), 
                                            train_preds)
    int_metrics = compute_metrics_lmc_gpr(model, 
                                          (split['test_int']['times'], split['test_int']['Y']), 
                                          int_preds)
    ext_metrics = compute_metrics_lmc_gpr(model, 
                                          (data['test_ext']['times'], data['test_ext']['Y']), 
                                          ext_preds)

    maes = dict(train=train_metrics['MAE'], 
                int=int_metrics['MAE'], 
                ext=ext_metrics['MAE'])

    nlpds = dict(train=train_metrics['NLPD'], 
                 int=int_metrics['NLPD'], 
                 ext=ext_metrics['NLPD'])

    print(f'\nMAE: {maes}')
    print(f'NLPD: {nlpds}\n')

    return (preds, maes, nlpds)

def train_IMC_SVGP(data, split_id, use_natgrad=True):
    split = data[f'split_{split_id}']
    P = split['train']['Y'].shape[0]
    L = 5
    kernel = 'Matern52'
    M = 30
    batch_size = 40

    with open(IND_PATH, 'rb') as fh:
        ind_times = pickle.load(fh)

    time_budget = relativedelta(days=1)

    train_params = dict(M=M, L=L, kernel_type=kernel, batch_size=batch_size,
                        n_steps=N_STEPS, lr=1e-3, gamma=1e-2, fix_ind=True, 
                        ind_init_mode='equal', ind_times=ind_times)

    model, elbos = train_imc_svgp(train_params, 
                                  split['train']['times'],
                                  split['train']['Y'],
                                  random_init=False,
                                  natgrad=use_natgrad,
                                  check_conv=True,
                                  check_budget=True,
                                  time_budget=time_budget)

    # Generate predictions on data
    train_preds = imc_svgp_predict(model, (split['train']['times'], split['train']['Y']))
    int_preds = imc_svgp_predict(model, (split['test_int']['times'], split['test_int']['Y']))
    ext_preds = imc_svgp_predict(model, (data['test_ext']['times'], data['test_ext']['Y']))

    preds = dict(train=train_preds, int=int_preds, ext=ext_preds)

    # Compute accuracy and uncertainty metrics
    train_metrics = compute_metrics_imc_svgp(model, 
                                             (split['train']['times'], split['train']['Y']), 
                                             train_preds)
    int_metrics = compute_metrics_imc_svgp(model, 
                                           (split['test_int']['times'], split['test_int']['Y']), 
                                           int_preds)
    ext_metrics = compute_metrics_imc_svgp(model, 
                                           (data['test_ext']['times'], data['test_ext']['Y']), 
                                           ext_preds)

    maes = dict(train=train_metrics['MAE'], 
                int=int_metrics['MAE'], 
                ext=ext_metrics['MAE'])

    nlpds = dict(train=train_metrics['NLPD'], 
                 int=int_metrics['NLPD'], 
                 ext=ext_metrics['NLPD'])

    print(f'\nMAE: {maes}')
    print(f'NLPD: {nlpds}\n')

    return (elbos, preds, maes, nlpds)

def train_LMC_SVGP(data, split_id, use_natgrad=True):
    split = data[f'split_{split_id}']
    P = split['train']['Y'].shape[0]
    L = 5
    kernel = 'Matern52'
    M = 30
    batch_size = 40

    with open(IND_PATH, 'rb') as fh:
        ind_times = pickle.load(fh)

    time_budget = relativedelta(days=1)

    train_params = dict(M=M, L=L, kernel_type=kernel, batch_size=batch_size,
                        n_steps=N_STEPS, lr=1e-3, gamma=1e-2, fix_ind=True, 
                        ind_init_mode='equal', ind_times=ind_times)

    model, elbos = train_lmc_svgp(train_params, 
                                  split['train']['times'],
                                  split['train']['Y'],
                                  random_init=False,
                                  natgrad=use_natgrad,
                                  check_conv=True,
                                  check_budget=True,
                                  time_budget=time_budget)

    # Generate predictions on data
    train_preds = lmc_svgp_predict(model, (split['train']['times'], split['train']['Y']))
    int_preds = lmc_svgp_predict(model, (split['test_int']['times'], split['test_int']['Y']))
    ext_preds = lmc_svgp_predict(model, (data['test_ext']['times'], data['test_ext']['Y']))

    preds = dict(train=train_preds, int=int_preds, ext=ext_preds)

    # Compute accuracy and uncertainty metrics
    train_metrics = compute_metrics_lmc_svgp(model, 
                                             (split['train']['times'], split['train']['Y']), 
                                             train_preds)
    int_metrics = compute_metrics_lmc_svgp(model, 
                                           (split['test_int']['times'], split['test_int']['Y']), 
                                           int_preds)
    ext_metrics = compute_metrics_lmc_svgp(model, 
                                           (data['test_ext']['times'], data['test_ext']['Y']), 
                                           ext_preds)

    maes = dict(train=train_metrics['MAE'], 
                int=int_metrics['MAE'], 
                ext=ext_metrics['MAE'])

    nlpds = dict(train=train_metrics['NLPD'], 
                 int=int_metrics['NLPD'], 
                 ext=ext_metrics['NLPD'])

    print(f'\nMAE: {maes}')
    print(f'NLPD: {nlpds}\n')

    return (elbos, preds, maes, nlpds)

def run_experiments(model_type, use_natgrad=True):
    """Trains baseline models on the small simulation data."""

    print(f'Running {N_FOLDS}-fold CV with model="{model_type}" (NGD={use_natgrad}).')

    # Load simulation data
    with open(DATA_PATH, 'rb') as fh:
        data = pickle.load(fh)

    results = defaultdict(list)

    # Train model
    if model_type == 'ind_gps':
        train_f = train_GPR

    elif model_type == 'lmc_gpr':
        train_f = train_LMC_GPR

    elif model_type == 'lmc_svgp':
        train_f = train_LMC_SVGP

    elif model_type == 'imc_svgp':
        train_f = train_IMC_SVGP

    for i in range(N_FOLDS):
        print(f'\nExperiment {i+1}/{N_FOLDS}:')
        exp_metrics = train_f(data, i+1, use_natgrad=use_natgrad)

        if len(exp_metrics) == 4:
            results['elbos'].append(exp_metrics[0])
            results['preds'].append(exp_metrics[1])
            results['maes'].append(exp_metrics[2])
            results['nlpds'].append(exp_metrics[3])

        else:
            results['preds'].append(exp_metrics[0])
            results['maes'].append(exp_metrics[1])
            results['nlpds'].append(exp_metrics[2])

    # Average and stddev on MAE and NLPD
    avg_mae, std_mae = mean_std(results['maes'])
    avg_nlpd, std_nlpd = mean_std(results['nlpds'])

    print(f'[Train MAE] {avg_mae["train"]: .5f} ± {std_mae["train"]: .5f}')
    print(f'[Train NLPD] {avg_nlpd["train"]: .5f} ± {std_nlpd["train"]: .5f}\n')

    print(f'[Int. MAE] {avg_mae["int"]: .5f} ± {std_mae["int"]: .5f}')
    print(f'[Int. NLPD] {avg_nlpd["int"]: .5f} ± {std_nlpd["int"]: .5f}\n')

    print(f'[Ext. MAE] {avg_mae["ext"]: .5f} ± {std_mae["ext"]: .5f}')
    print(f'[Ext. NLPD] {avg_nlpd["ext"]: .5f} ± {std_nlpd["ext"]: .5f}\n')

    return results

def main(**kwargs):
    lmc_gpr_flag = kwargs['lmc_gpr']
    lmc_svgp_flag = kwargs['lmc_svgp']
    imc_svgp_flag = kwargs['imc_svgp']
    natgrad = kwargs['natgrad']

    if not osp.isdir(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)

    if lmc_gpr_flag:
        model_type = 'lmc_gpr'
        lmc_gpr_results = run_experiments(model_type, use_natgrad=None)
        save_results(model_type, lmc_gpr_results, use_natgrad=None)

    if lmc_svgp_flag:
        model_type = 'lmc_svgp'

        if natgrad:
            natgrad_results = run_experiments(model_type)
            save_results(model_type, natgrad_results, use_natgrad=True)
        
        else:
            adam_results = run_experiments(model_type, use_natgrad=False)
            save_results(model_type, adam_results, use_natgrad=False)

    if imc_svgp_flag:
        model_type = 'imc_svgp'
        
        if natgrad:
            natgrad_results = run_experiments(model_type)
            save_results(model_type, natgrad_results, use_natgrad=True)
        
        else:
            adam_results = run_experiments(model_type, use_natgrad=False)
            save_results(model_type, adam_results, use_natgrad=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmc_gpr', default=False, help='Option to train LMC-GPR', action='store_true')
    parser.add_argument('--lmc_svgp', default=False, help='Option to train LMC-SVGP', action='store_true')
    parser.add_argument('--imc_svgp', default=False, help='Option to train IMC-SVGP', action='store_true')
    parser.add_argument('--natgrad', default=False, help='Option to use natural gradients', action='store_true')
    args = parser.parse_args()

    main(**vars(args))
