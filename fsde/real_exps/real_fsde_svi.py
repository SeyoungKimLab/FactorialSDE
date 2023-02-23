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

ID_TO_DATASET = ['covid', 'stock', 'energy', 'air_quality']

ID_TO_DATA = ['../../data/covid_data.pkl',
              '../../data/stock_data.pkl',
              '../../data/energy_data.pkl',
              '../../data/air_quality_data.pkl']

N_STEPS = 50000 # Max iteration

ID_TO_LR = [
    dict(model_lr=1e-3, var_adam_lr=1e-4, var_lr_init=1e-5, var_lr_end=1e-4),
    dict(model_lr=1e-3, var_adam_lr=1e-3, var_lr_init=1e-5, var_lr_end=1e-3),
    dict(model_lr=1e-3, var_adam_lr=1e-3, var_lr_init=1e-5, var_lr_end=1e-3),
    dict(model_lr=1e-3, var_adam_lr=1e-3, var_lr_init=1e-5, var_lr_end=1e-3)
]

ID_TO_M = [50,500,1000,2000]
ID_TO_IND_PATHS = [
    './ind_times_covid.pkl',
    './ind_times_stock.pkl',
    './ind_times_energy.pkl',
    './ind_times_air_quality.pkl'
]
ID_TO_LRSTEPS = [15000,1000,1000,1000]
ID_TO_LATENT = [300,15,5,5]
ID_TO_BATCH_SIZE = [150,500,1000,1000]
JITTER = 1e-8

def train_FSDE_SVI(data, data_id, use_natgrad=True):
    train_data = Dataset(times=jnp.asarray(data['train']['times']), 
                         Y=jnp.asarray(data['train']['Y']))
    P = train_data.P
    L = ID_TO_LATENT[data_id]
    kernel = 'Matern32'
    M = ID_TO_M[data_id]
    batch_size = ID_TO_BATCH_SIZE[data_id]

    # Initialize parameters using seed
    init_tuple = init_params(kernel=kernel, L=L, P=P, M=M,
                             var=0.1, lengthscale=1.)

    model_params, v_params, compute_cov_infty, compute_F = init_tuple

    with open(ID_TO_IND_PATHS[data_id], 'rb') as fh:
        ind_times = pickle.load(fh)

    # Train model
    model = FSDE_SVI(model_params=model_params,
                     v_params=v_params,
                     kernel=kernel,
                     jitter=JITTER,
                     ind_times=ind_times,
                     compute_cov_infty=compute_cov_infty,
                     compute_F=compute_F,
                     num_times=train_data.T)

    lr = ID_TO_LR[data_id]

    exp_metrics = model.fit(train_dataset=train_data,
                            n_steps=N_STEPS,
                            batch_size=batch_size,
                            key=jr.PRNGKey(0),
                            lr=lr,
                            lr_steps=ID_TO_LRSTEPS[data_id],
                            use_natgrad=use_natgrad)
    elbos = exp_metrics[0]

    # Generate predictions on data
    pred_args = model.precompute_pred_args()
    Y_int_hat, Y_int_cov = model.predict(data['test_int']['times'].squeeze(), *pred_args)
    Y_ext_hat, Y_ext_cov = model.predict(data['test_ext']['times'].squeeze(), *pred_args)

    preds = dict(int=(Y_int_hat, Y_int_cov),
                 ext=(Y_ext_hat, Y_ext_cov))

    # Compute accuracy and uncertainty metrics
    int_mae = compute_MAE(data['test_int']['Y'], Y_int_hat)
    int_nlpd = FSDE_NLPD(data['test_int']['Y'], Y_int_hat, Y_int_cov)
    ext_mae = compute_MAE(data['test_ext']['Y'], Y_ext_hat)
    ext_nlpd = FSDE_NLPD(data['test_ext']['Y'], Y_ext_hat, Y_ext_cov)

    maes = dict(int=int_mae, ext=ext_mae)
    nlpds = dict(int=int_nlpd, ext=ext_nlpd)

    print(f'\nMAE: {maes}')
    print(f'NLPD: {nlpds}\n')

    return (elbos, preds, maes, nlpds)

def run_experiment(data_id, use_natgrad=True):
    """Trains FSDE-SVI on real data."""

    dataset = ID_TO_DATASET[data_id]
    print(f'Running FSDE-SVI on {dataset.upper()} (NGD={use_natgrad}).')

    # Load simulation data
    with open(ID_TO_DATA[data_id], 'rb') as fh:
        data = pickle.load(fh)

    # Train model
    elbos, preds, maes, nlpds = train_FSDE_SVI(data, data_id, use_natgrad=use_natgrad)

    print(f'[Int. MAE] {maes["int"]: .5f}')
    print(f'[Int. NLPD] {nlpds["int"]: .5f}\n')

    print(f'[Ext. MAE] {maes["ext"]: .5f}')
    print(f'[Ext. NLPD] {nlpds["ext"]: .5f}\n')

    # Don't save predictions if using COVID data (too large)
    if data_id == 0:
        results = dict(elbos=elbos, maes=maes, nlpds=nlpds)
    else:
        results = dict(elbos=elbos, preds=preds, maes=maes, nlpds=nlpds)
    
    results_dir = f'./{dataset}_results'
    pkl_str = f'svi_results_{dataset}'
    pkl_str += '_natgrad.pkl' if use_natgrad else '_adam.pkl'
    pkl_path = osp.join(results_dir, pkl_str)

    with open(pkl_path, 'wb') as fh:
        pickle.dump(results, fh)

    print(f'[{dataset.upper()}] FSDE-SVI results saved: {pkl_path}.\n')

def main(**kwargs):
    data_id = int(kwargs['data_id'])-1
    natgrad = kwargs['natgrad']

    results_dir = f'./{ID_TO_DATASET[data_id]}_results'
    if not osp.isdir(results_dir):
        os.mkdir(results_dir)

    if natgrad:
        natgrad_results = run_experiment(data_id, use_natgrad=True)
    else:
        adam_results = run_experiment(data_id, use_natgrad=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_id', help='Specifies dataset to use')
    parser.add_argument('--natgrad', default=False, help='Option to use natural gradients', action='store_true')
    args = parser.parse_args()

    main(**vars(args))
