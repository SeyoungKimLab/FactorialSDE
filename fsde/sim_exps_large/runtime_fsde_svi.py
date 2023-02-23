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

RUNTIME_DIR = './runtime_results'
DATA_PATH = '../../data/sim_large_data.pkl'
IND_PATH = './ind_times_{}.pkl'
JITTER = 1e-8
N_STEPS = 50000
ID_TO_M = [200,400,600,800]

def train_FSDE_SVI(data, ind_id, split_id, use_natgrad=True):
    split = data[f'split_{split_id}']
    split_train = Dataset(times=jnp.asarray(split['train']['times']), 
                          Y=jnp.asarray(split['train']['Y']))
    P = split_train.P
    L = 10
    kernel = 'Matern32'
    M = ID_TO_M[ind_id-1]
    batch_size = 1000

    # Initialize parameters using seed
    init_tuple = init_params(kernel=kernel, L=L, P=P, M=M,
                             var=0.1, lengthscale=1.)

    model_params, v_params, compute_cov_infty, compute_F = init_tuple
    
    with open(IND_PATH.format(M), 'rb') as fh:
        ind_times = pickle.load(fh)

    # Train model
    model = FSDE_SVI(model_params=model_params,
                     v_params=v_params,
                     kernel=kernel,
                     jitter=JITTER,
                     ind_times=ind_times,
                     compute_cov_infty=compute_cov_infty,
                     compute_F=compute_F,
                     num_times=split_train.T)

    lr = dict(model_lr=1e-2,
              var_adam_lr=1e-2,
              var_lr_init=1e-4,
              var_lr_end=1e-2)

    exp_metrics = model.fit(train_dataset=split_train,
                            n_steps=N_STEPS,
                            batch_size=batch_size,
                            key=jr.PRNGKey(0),
                            lr=lr,
                            lr_steps=500,
                            use_natgrad=use_natgrad)
    elbos = exp_metrics[0]

    # Generate predictions on data
    pred_args = model.precompute_pred_args()
    Y_int_hat, Y_int_cov = model.predict(split['test_int']['times'].squeeze(), *pred_args)
    Y_ext_hat, Y_ext_cov = model.predict(data['test_ext']['times'].squeeze(), *pred_args)

    preds = dict(int=(Y_int_hat, Y_int_cov),
                 ext=(Y_ext_hat, Y_ext_cov))

    # Compute accuracy and uncertainty metrics
    int_mae = compute_MAE(split['test_int']['Y'], Y_int_hat)
    int_nlpd = FSDE_NLPD(split['test_int']['Y'], Y_int_hat, Y_int_cov)
    ext_mae = compute_MAE(data['test_ext']['Y'], Y_ext_hat)
    ext_nlpd = FSDE_NLPD(data['test_ext']['Y'], Y_ext_hat, Y_ext_cov)

    maes = dict(int=int_mae, ext=ext_mae)
    nlpds = dict(int=int_nlpd, ext=ext_nlpd)

    print(f'\nMAE: {maes}')
    print(f'NLPD: {nlpds}\n')

    return (elbos, preds, maes, nlpds)

def run_experiment(ind_id, split_id, use_natgrad=True):
    """Trains FSDE-SVI on large simulation data with different number of inducing points."""

    M = ID_TO_M[ind_id-1]
    print(f'Running FSDE-SVI on fold {split_id} (M={M}, NGD={use_natgrad}).')

    # Load simulation data
    with open(DATA_PATH, 'rb') as fh:
        data = pickle.load(fh)

    # Train model
    elbos, preds, maes, nlpds = train_FSDE_SVI(data, ind_id, split_id, use_natgrad=use_natgrad)

    print(f'[Int. MAE] {maes["int"]: .5f}')
    print(f'[Int. NLPD] {nlpds["int"]: .5f}\n')

    print(f'[Ext. MAE] {maes["ext"]: .5f}')
    print(f'[Ext. NLPD] {nlpds["ext"]: .5f}\n')

    results = dict(elbos=elbos, preds=preds, maes=maes, nlpds=nlpds)

    pkl_str = f'svi_results_{split_id}_{M}'
    pkl_str += '_natgrad.pkl' if use_natgrad else '_adam.pkl'
    pkl_path = osp.join(RUNTIME_DIR, pkl_str)

    with open(pkl_path, 'wb') as fh:
        pickle.dump(results, fh)

    print(f'[Runtime] FSDE-SVI results saved: {pkl_path}.\n')

def main(**kwargs):
    split_id = int(kwargs['split_id'])
    ind_id = int(kwargs['ind_id'])
    natgrad = kwargs['natgrad']

    if not osp.isdir(RUNTIME_DIR):
        os.mkdir(RUNTIME_DIR)

    if natgrad:
        natgrad_results = run_experiment(ind_id, split_id, use_natgrad=True)
    else:
        adam_results = run_experiment(ind_id, split_id, use_natgrad=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ind_id', help='Specifies number of inducing points to use')
    parser.add_argument('--split_id', default=1, help='Specifies the split to use')
    parser.add_argument('--natgrad', default=False, help='Option to use natural gradients', action='store_true')
    args = parser.parse_args()

    main(**vars(args))