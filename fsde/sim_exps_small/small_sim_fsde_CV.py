import sys
sys.path.append('../../')

import os
import os.path as osp
import argparse
from collections import defaultdict
import pickle
import chex
import typing
import jaxtyping
from jaxtyping import f64, i64

import jax
jax.config.update("jax_enable_x64", True) # For setting default dtype to float64
import jax.numpy as jnp
import jax.random as jr
from jax import grad, jit, vmap, pmap, tree_map, lax, value_and_grad
from jax.tree_util import Partial
import jax.scipy as jscipy
from jax.scipy.linalg import expm
jnp.set_printoptions(suppress=True)

from functools import partial
from tqdm import tqdm

import time
import numpy as np
import copy
import scipy
from sklearn.model_selection import train_test_split

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

import optax

import fsde
from fsde.core.ops import *
from fsde.core.model_utils import *
from fsde.models import FSDE, FSDE_SVI

# CONSTANTS

RESULTS_DIR = './sim_results'
DATA_PATH = '../../data/sim_small_data.pkl'
IND_PATH = './ind_times.pkl'
JITTER = 1e-8
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

    train_dataset = Dataset(times=jnp.asarray(train_times), Y=jnp.asarray(Y_train))
    int_dataset = Dataset(times=jnp.asarray(int_times), Y=jnp.asarray(Y_int))
    ext_dataset = Dataset(times=jnp.asarray(ext_times), Y=jnp.asarray(Y_ext))
    datasets = dict(train=train_dataset, int=int_dataset, ext=ext_dataset)
    idxs = dict(train=train_idx, int=int_idx, ext=ext_idx)

    return datasets, idxs

def mean_std(metrics):
    train_metrics = jnp.array([metric['train'] for metric in metrics])
    int_metrics = jnp.array([metric['int'] for metric in metrics])
    ext_metrics = jnp.array([metric['ext'] for metric in metrics])

    metrics = dict(train=train_metrics, int=int_metrics, ext=ext_metrics)
    avg = tree_map(jnp.mean, metrics)
    std = tree_map(jnp.std, metrics)

    return avg, std

def save_results(model_type, results, use_natgrad):
    if model_type == 'fsde':
        pkl_path = f'fsde_results_CV_natgrad.pkl' if use_natgrad else f'fsde_results_CV_adam.pkl'

    elif model_type == 'svi':
        pkl_path = f'svi_results_CV_natgrad.pkl' if use_natgrad else f'svi_results_CV_adam.pkl'

    pkl_path = osp.join(RESULTS_DIR, pkl_path)
    with open(pkl_path, 'wb') as fh:
        pickle.dump(results, fh)

    print(f'[Small Simulation] {model_type} results saved: {pkl_path}.\n')

def train_FSDE(data, split_id, use_natgrad=True):
    split = data[f'split_{split_id}']
    split_train = Dataset(times=jnp.asarray(split['train']['times']), 
                          Y=jnp.asarray(split['train']['Y']))
    P = split_train.P
    L = 5
    kernel = 'Matern52'

    # Initialize parameters using seed
    init_tuple = init_params(kernel=kernel, L=L, P=P, M=split_train.T,
                             var=0.1, lengthscale=1.)

    model_params, v_params, compute_cov_infty, compute_F = init_tuple

    # Train model
    model = FSDE(model_params=model_params, 
                 v_params=v_params, 
                 kernel=kernel,
                 train_dataset=split_train, 
                 jitter=JITTER,
                 compute_cov_infty=compute_cov_infty, 
                 compute_F=compute_F)

    lr = dict(model_lr=1e-3,
              var_adam_lr=1e-3,
              var_lr_init=1e-4,
              var_lr_end=1e-2)

    exp_metrics = model.fit(n_steps=N_STEPS, lr=lr, lr_steps=4000, use_natgrad=use_natgrad)
    elbos = exp_metrics[0]

    # Generate predictions on data
    pred_args = model.precompute_pred_args()
    Y_train_hat, Y_train_cov = model.predict_train()
    Y_int_hat, Y_int_cov = model.predict(split['test_int']['times'].squeeze(), *pred_args)
    Y_ext_hat, Y_ext_cov = model.predict(data['test_ext']['times'].squeeze(), *pred_args)

    preds = dict(train=(Y_train_hat, Y_train_cov),
                 int=(Y_int_hat, Y_int_cov),
                 ext=(Y_ext_hat, Y_ext_cov))

    # Compute accuracy and uncertainty metrics
    train_mae = compute_MAE(split['train']['Y'], Y_train_hat)
    train_nlpd = FSDE_NLPD(split['train']['Y'], Y_train_hat, Y_train_cov)
    int_mae = compute_MAE(split['test_int']['Y'], Y_int_hat)
    int_nlpd = FSDE_NLPD(split['test_int']['Y'], Y_int_hat, Y_int_cov)
    ext_mae = compute_MAE(data['test_ext']['Y'], Y_ext_hat)
    ext_nlpd = FSDE_NLPD(data['test_ext']['Y'], Y_ext_hat, Y_ext_cov)

    maes = dict(train=train_mae, int=int_mae, ext=ext_mae)
    nlpds = dict(train=train_nlpd, int=int_nlpd, ext=ext_nlpd)

    print(f'\nMAE: {maes}')
    print(f'NLPD: {nlpds}\n')

    return (elbos, preds, maes, nlpds)

def train_FSDE_SVI(data, split_id, use_natgrad=True):
    split = data[f'split_{split_id}']
    split_train = Dataset(times=jnp.asarray(split['train']['times']), 
                          Y=jnp.asarray(split['train']['Y']))
    P = split_train.P
    L = 5
    kernel = 'Matern52'
    M = 30
    batch_size = 40

    # Initialize parameters using seed
    init_tuple = init_params(kernel=kernel, L=L, P=P, M=M,
                             var=0.1, lengthscale=1.)

    model_params, v_params, compute_cov_infty, compute_F = init_tuple

    with open(IND_PATH, 'rb') as fh:
        ind_times = jnp.asarray(pickle.load(fh)).squeeze()

    # Train model
    model = FSDE_SVI(model_params=model_params,
                     v_params=v_params,
                     kernel=kernel,
                     jitter=JITTER,
                     ind_times=ind_times,
                     compute_cov_infty=compute_cov_infty,
                     compute_F=compute_F,
                     num_times=split_train.T)

    lr = dict(model_lr=1e-3,
              var_adam_lr=1e-3,
              var_lr_init=1e-5,
              var_lr_end=1e-4)

    exp_metrics = model.fit(train_dataset=split_train,
                            n_steps=N_STEPS,
                            batch_size=40,
                            key=jr.PRNGKey(0),
                            lr=lr,
                            lr_steps=4000,
                            use_natgrad=use_natgrad)
    elbos = exp_metrics[0]

    # Generate predictions on data
    pred_args = model.precompute_pred_args()
    Y_train_hat, Y_train_cov = model.predict(split['train']['times'].squeeze(), *pred_args)
    Y_int_hat, Y_int_cov = model.predict(split['test_int']['times'].squeeze(), *pred_args)
    Y_ext_hat, Y_ext_cov = model.predict(data['test_ext']['times'].squeeze(), *pred_args)

    preds = dict(train=(Y_train_hat, Y_train_cov),
                 int=(Y_int_hat, Y_int_cov),
                 ext=(Y_ext_hat, Y_ext_cov))

    # Compute accuracy and uncertainty metrics
    train_mae = compute_MAE(split['train']['Y'], Y_train_hat)
    train_nlpd = FSDE_NLPD(split['train']['Y'], Y_train_hat, Y_train_cov)
    int_mae = compute_MAE(split['test_int']['Y'], Y_int_hat)
    int_nlpd = FSDE_NLPD(split['test_int']['Y'], Y_int_hat, Y_int_cov)
    ext_mae = compute_MAE(data['test_ext']['Y'], Y_ext_hat)
    ext_nlpd = FSDE_NLPD(data['test_ext']['Y'], Y_ext_hat, Y_ext_cov)

    maes = dict(train=train_mae, int=int_mae, ext=ext_mae)
    nlpds = dict(train=train_nlpd, int=int_nlpd, ext=ext_nlpd)

    print(f'\nMAE: {maes}')
    print(f'NLPD: {nlpds}\n')

    return (elbos, preds, maes, nlpds)

def run_experiments(model_type, use_natgrad=True):
    """Trains FSDE and FSDE-SVI models on the small simulation data."""

    print(f'Running {N_FOLDS}-fold CV with model="{model_type}" (NGD={use_natgrad}).')

    # Load simulation data
    with open(DATA_PATH, 'rb') as fh:
        data = pickle.load(fh)

    results = defaultdict(list)

    # Train model with multiple initializations
    train_f = train_FSDE if model_type == 'fsde' else train_FSDE_SVI

    for i in range(N_FOLDS):
        print(f'\nExperiment {i+1}/{N_FOLDS}:')
        elbos, preds, maes, nlpds = train_f(data, i+1, use_natgrad=use_natgrad)
        
        results['elbos'].append(elbos)
        results['preds'].append(preds)
        results['maes'].append(maes)
        results['nlpds'].append(nlpds)

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
    fsde_flag = kwargs['fsde']
    svi_flag = kwargs['svi']
    natgrad = kwargs['natgrad']

    if not osp.isdir(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)

    if fsde_flag:
        model_type = 'fsde'

        if natgrad:
            natgrad_results = run_experiments(model_type)
            save_results(model_type, natgrad_results, use_natgrad=True)
        
        else:
            adam_results = run_experiments(model_type, use_natgrad=False)
            save_results(model_type, adam_results, use_natgrad=False)

    if svi_flag:
        model_type = 'svi'
        
        if natgrad:
            natgrad_results = run_experiments(model_type)
            save_results(model_type, natgrad_results, use_natgrad=True)
        
        else:
            adam_results = run_experiments(model_type, use_natgrad=False)
            save_results(model_type, adam_results, use_natgrad=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fsde', default=False, help='Option to train FSDE', action='store_true')
    parser.add_argument('--svi', default=False, help='Option to train FSDE-SVI', action='store_true')
    parser.add_argument('--natgrad', default=False, help='Option to use natural gradients', action='store_true')
    args = parser.parse_args()

    main(**vars(args))
