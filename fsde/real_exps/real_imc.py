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

ID_TO_DATASET = ['covid', 'stock', 'energy', 'air_quality']

ID_TO_DATA = ['../../data/covid_data.pkl',
              '../../data/stock_data.pkl',
              '../../data/energy_data.pkl',
              '../../data/air_quality_data.pkl']

N_STEPS = 50000

ID_TO_LR = [(1e-3,1e-2),(1e-3,1e-2),(1e-3,1e-2),(1e-3,1e-2)]

ID_TO_M = [50,500,1000,2000]
ID_TO_IND_PATHS = [
    './ind_times_covid.pkl',
    './ind_times_stock.pkl',
    './ind_times_energy.pkl',
    './ind_times_air_quality.pkl'
]
ID_TO_LATENT = [300,15,5,5]
ID_TO_BATCH_SIZE = [150,500,1000,1000]

def train_IMC_SVGP(data, data_id, use_natgrad=True):
    P = data['train']['Y'].shape[0]
    L = ID_TO_LATENT[data_id]
    kernel = 'Matern32'
    M = ID_TO_M[data_id]
    batch_size = ID_TO_BATCH_SIZE[data_id]

    with open(ID_TO_IND_PATHS[data_id], 'rb') as fh:
        ind_times = pickle.load(fh)
    
    lr = ID_TO_LR[data_id]
    time_budget = relativedelta(days=1)

    train_params = dict(M=M, L=L, kernel_type=kernel, batch_size=batch_size,
                        n_steps=N_STEPS, lr=lr[0], gamma=lr[1], fix_ind=True, 
                        ind_init_mode='equal', ind_times=ind_times)

    model, elbos = train_imc_svgp(train_params, 
                                  data['train']['times'],
                                  data['train']['Y'],
                                  random_init=False,
                                  natgrad=use_natgrad,
                                  check_conv=True,
                                  check_budget=True,
                                  time_budget=time_budget)

    # Generate predictions on data
    int_preds = imc_svgp_predict(model, (data['test_int']['times'], data['test_int']['Y']))
    ext_preds = imc_svgp_predict(model, (data['test_ext']['times'], data['test_ext']['Y']))

    preds = dict(int=int_preds, ext=ext_preds)

    # Compute accuracy and uncertainty metrics
    int_metrics = compute_metrics_imc_svgp(model, 
                                           (data['test_int']['times'], data['test_int']['Y']), 
                                           int_preds)
    ext_metrics = compute_metrics_imc_svgp(model, 
                                           (data['test_ext']['times'], data['test_ext']['Y']), 
                                           ext_preds)

    maes = dict(int=int_metrics['MAE'], 
                ext=ext_metrics['MAE'])

    nlpds = dict(int=int_metrics['NLPD'], 
                 ext=ext_metrics['NLPD'])

    print(f'\nMAE: {maes}')
    print(f'NLPD: {nlpds}\n')

    return (elbos, preds, maes, nlpds)

def run_experiment(data_id, use_natgrad=True):
    """Trains IMC on real data."""

    dataset = ID_TO_DATASET[data_id]
    print(f'Running IMC on {dataset.upper()} (NGD={use_natgrad}).')

    # Load simulation data
    with open(ID_TO_DATA[data_id], 'rb') as fh:
        data = pickle.load(fh)

    # Train model
    elbos, preds, maes, nlpds = train_IMC_SVGP(data, data_id, use_natgrad=use_natgrad)

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
    pkl_str = f'imc_svgp_results_{dataset}'
    pkl_str += '_natgrad.pkl' if use_natgrad else '_adam.pkl'
    pkl_path = osp.join(results_dir, pkl_str)

    with open(pkl_path, 'wb') as fh:
        pickle.dump(results, fh)

    print(f'[{dataset.upper()}] IMC results saved: {pkl_path}.\n')

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