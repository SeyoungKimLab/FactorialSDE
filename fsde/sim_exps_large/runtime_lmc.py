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

RUNTIME_DIR = './runtime_results'
DATA_PATH = '../../data/sim_large_data.pkl'
IND_PATH = './ind_times_{}.pkl'
N_STEPS = 50000
ID_TO_M = [200,400,600,800]

def train_LMC_SVGP(data, ind_id, split_id, use_natgrad=True):
    split = data[f'split_{split_id}']
    P = split['train']['Y'].shape[0]
    L = 10
    kernel = 'Matern32'
    M = ID_TO_M[ind_id-1]
    batch_size = 1000

    with open(IND_PATH.format(M), 'rb') as fh:
        ind_times = pickle.load(fh)

    time_budget = relativedelta(days=1)

    train_params = dict(M=M, L=L, kernel_type=kernel, batch_size=batch_size,
                        n_steps=N_STEPS, lr=1e-2, gamma=1e-2, fix_ind=True, 
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
    int_preds = lmc_svgp_predict(model, (split['test_int']['times'], split['test_int']['Y']))
    ext_preds = lmc_svgp_predict(model, (data['test_ext']['times'], data['test_ext']['Y']))

    preds = dict(int=int_preds, ext=ext_preds)

    # Compute accuracy and uncertainty metrics
    int_metrics = compute_metrics_lmc_svgp(model, 
                                           (split['test_int']['times'], split['test_int']['Y']), 
                                           int_preds)
    ext_metrics = compute_metrics_lmc_svgp(model, 
                                           (data['test_ext']['times'], data['test_ext']['Y']), 
                                           ext_preds)

    maes = dict(int=int_metrics['MAE'], 
                ext=ext_metrics['MAE'])

    nlpds = dict(int=int_metrics['NLPD'], 
                 ext=ext_metrics['NLPD'])

    print(f'\nMAE: {maes}')
    print(f'NLPD: {nlpds}\n')

    return (elbos, preds, maes, nlpds)

def run_experiment(ind_id, split_id, use_natgrad=True):
    """Trains LMC on large simulation data with different number of inducing points."""

    M = ID_TO_M[ind_id-1]
    print(f'Running LMC on fold {split_id} (M={M}, NGD={use_natgrad}).')

    # Load simulation data
    with open(DATA_PATH, 'rb') as fh:
        data = pickle.load(fh)

    # Train model
    elbos, preds, maes, nlpds = train_LMC_SVGP(data, ind_id, split_id, use_natgrad=use_natgrad)

    print(f'[Int. MAE] {maes["int"]: .5f}')
    print(f'[Int. NLPD] {nlpds["int"]: .5f}\n')

    print(f'[Ext. MAE] {maes["ext"]: .5f}')
    print(f'[Ext. NLPD] {nlpds["ext"]: .5f}\n')

    results = dict(elbos=elbos, preds=preds, maes=maes, nlpds=nlpds)
    
    pkl_str = f'lmc_svgp_results_{split_id}_{M}'
    pkl_str += '_natgrad.pkl' if use_natgrad else '_adam.pkl'
    pkl_path = osp.join(RUNTIME_DIR, pkl_str)

    with open(pkl_path, 'wb') as fh:
        pickle.dump(results, fh)

    print(f'[Runtime] LMC results saved: {pkl_path}.\n')

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
