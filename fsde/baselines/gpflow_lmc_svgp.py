import sys
sys.path.append('../../')

import os
import os.path as osp
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import scipy
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

from sklearn.cluster import KMeans

def vec(A):
    """Vectorizes a given matrix."""
    return tf.reshape(tf.transpose(A), (-1,1)) # [P,T] -> [PT,1]

def quadratic(A, B, jitter=1e-8):
    """Computes A^T B^{-1} A, assuming B is PD."""

    d = B.shape[-1]
    L = np.linalg.cholesky(B + jitter * np.eye(d))
    S = scipy.linalg.solve_triangular(L, A, lower=True)

    return S.T @ S

def ind_kmeans(M, data):
    """Runs K-means on input points for inducing input initialization."""

    if data.ndim == 1:
        data = data[:,None]

    kmeans = KMeans(n_clusters=M).fit(data)
    ind_points = kmeans.cluster_centers_

    return ind_points

def ind_equal(M, data, margin=0.15):
    """Returns a set of equally spaced input points for inducing input initialization."""

    data = data.squeeze()
    first = data[0] - margin
    last = data[-1] + margin
    ind_times = np.linspace(first, last, M)[:,None]

    return ind_times

def NLPD(Y, Y_mean, Y_cov):
    """
    Computes the negative log-predictive density.

    Shapes: Y: [P,T], Y_mean: [P,T], Y_cov: [T,P,P]
    """

    P, T = Y.shape

    def _nlpd(pred_i):
        """Expects Y_i: [P,], Y_mean_i: [P,], Y_cov_i: [P,P]"""

        # Unpack
        Y_i, Y_mean_i, Y_cov_i = pred_i
        
        if Y_i.ndim == 1:
            Y_i = Y_i[:,None] # [P,1]

        if Y_mean_i.ndim == 1:
            Y_mean_i = Y_mean_i[:,None] # [P,1]

        logdet_cov_i = np.multiply(*np.linalg.slogdet(Y_cov_i))
        residual = Y_i - Y_mean_i

        nlpd = 0.5 * P * np.log(2 * np.pi)
        nlpd += 0.5 * logdet_cov_i
        nlpd += 0.5 * quadratic(residual, Y_cov_i).squeeze()

        return nlpd

    zipped = list(zip(Y.T, Y_mean.T, Y_cov))
    nlpds = list(map(_nlpd, zipped))

    return (1./(P*T)) * np.sum(nlpds)

def get_lmc_svgp(
    lmc_params, 
    train_times, 
    fix_ind=False, 
    random_init=False,
    seed=0, 
    verbose=False, 
    margin=0.15,
    ind_times=None
):
    """Returns a GPflow SVGP model with LMC kernel."""

    M = lmc_params['M'] # Number of inducing points
    T = train_times.shape[0] # Number of time points
    P = lmc_params['P'] # Number of outputs
    L = lmc_params['L'] # Number of latent GPs
    kernel_type = lmc_params['kernel_type'] # Type of kernel function
    ind_init_mode = lmc_params['ind_init_mode']

    # Define LMC kernel
    if kernel_type == 'SE':
        kernel_list = [SquaredExponential() for _ in range(L)]

    elif kernel_type == 'Matern32':
        kernel_list = [Matern32() for _ in range(L)]

    elif kernel_type == 'Matern52':
        kernel_list = [Matern52() for _ in range(L)]

    np.random.seed(seed)
    W = np.random.randn(P,L)
    kernel = LinearCoregionalization(kernel_list, W=W)

    # Initialize inducing points
    if ind_times is None:
        if ind_init_mode == 'K-means':
            Z = ind_kmeans(M, train_times)
        else:
            Z = ind_equal(M, train_times, margin=margin)
    else:
        Z = ind_times

    ind_points = SharedIndependentInducingVariables(InducingPoints(Z))

    # Initialize variational parameters
    if random_init:
        np.random.seed(seed) # Set random seed
        q_mu = np.random.normal(size=(M,L))
        A = np.random.normal(size=(M,M))
        B = A @ A.T + 1e-5 * np.eye(M)
        chol = np.linalg.cholesky(B)
        q_sqrt = np.repeat(chol[None,:], L, axis=0) # [L,M,M]

    else:
        q_mu = np.zeros((M,L))
        q_sqrt = np.repeat(np.eye(M)[None,:], L, axis=0) # [L,M,M]

    # Define model
    model = SVGP(kernel, Gaussian(), num_data=T, inducing_variable=ind_points, q_mu=q_mu, q_sqrt=q_sqrt)

    if fix_ind:
        set_trainable(model.inducing_variable.inducing_variable.Z, False)

    if verbose:
        print_summary(model)

    return model

def run_adam(
    model, 
    train_dataset, 
    n, 
    batch_size, 
    n_steps, 
    lr=1e-3,
    check_conv=False,
    window=40,
    tol=1e-5,
    patience=5,
    check_budget=False,
    start_time=None,
    time_budget=None
):
    """Runs the ADAM optimizer for specified iterations."""

    elbos = []
    #n_steps_per_epoch = n // batch_size
    patience_cnt = 0
    train_iter = iter(train_dataset.batch(batch_size))
    training_loss = model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.optimizers.Adam(learning_rate=lr)

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    '''
    with tqdm(range(n_epochs), unit='epoch') as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f'Epoch {epoch+1}')
            #epoch_loss = tf.convert_to_tensor(tf.Variable(0, dtype=default_float()))
            epoch_id = epoch + 1

            for step in range(n_steps_per_epoch):
                optimization_step()
                #epoch_loss += training_loss().numpy()
                step_elbo = -training_loss().numpy()
                elbos.append(step_elbo)
                tepoch.set_postfix(ELBO=step_elbo)
                
            #epoch_loss = epoch_loss / n_steps_per_epoch # Take average over steps in epoch
            #elbos.append(-epoch_loss)
    '''

    with tqdm(range(n_steps), unit='steps') as tstep:
        for step in tstep:
            tstep.set_description(f'Step {step+1}')
            optimization_step()
            step_elbo = -training_loss().numpy()
            elbos.append(step_elbo)
            tstep.set_postfix(ELBO=step_elbo)

            if check_conv:
                if (step+1) >= 2*window and (step+1) % window == 0:
                    prev_avg = np.mean(elbos[-2*window:-window])
                    curr_avg = np.mean(elbos[-window:])
                    p_change = np.abs((curr_avg - prev_avg) / prev_avg)

                    if p_change <= tol:
                        patience_cnt += 1
                        print(f'Remaining patience: {patience - patience_cnt}')

                    if patience_cnt >= patience:
                        print('Model convergence. Stopping training.')
                        break

            if check_budget:
                current_time = datetime.now()

                if current_time >= start_time + time_budget:
                    print('Time budget reached. Stopping training.')
                    break

    return elbos

def run_adam_natgrad(
    model, 
    train_dataset, 
    n, 
    batch_size, 
    n_steps, 
    lr=1e-3, 
    gamma=1e-2,
    check_conv=False,
    window=40,
    tol=1e-5,
    patience=5,
    check_budget=False,
    start_time=None,
    time_budget=None
):
    """Runs the ADAM + NatGrad optimizers for specified iterations."""

    elbos = []
    #n_steps_per_epoch = n // batch_size
    patience_cnt = 0
    train_iter = iter(train_dataset.batch(batch_size))
    training_loss = model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.optimizers.Adam(learning_rate=lr)
    ng_optimizer = gpflow.optimizers.NaturalGradient(gamma=gamma)

    # Stop ADAM from training variational parameters
    set_trainable(model.q_mu, False)
    set_trainable(model.q_sqrt, False)
    variational_params = [(model.q_mu, model.q_sqrt)]

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)
        ng_optimizer.minimize(training_loss, variational_params)

    '''
    with tqdm(range(n_epochs), unit='epoch') as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f'Epoch {epoch+1}')
            #epoch_loss = tf.convert_to_tensor(tf.Variable(0, dtype=default_float()))
            epoch_id = epoch + 1

            for step in range(n_steps_per_epoch):
                optimization_step()
                #epoch_loss += training_loss().numpy()
                step_elbo = -training_loss().numpy()
                elbos.append(step_elbo)
                tepoch.set_postfix(ELBO=step_elbo)

            #epoch_loss = epoch_loss / n_steps_per_epoch # Take average over steps in epoch
            #elbos.append(-epoch_loss)
    '''

    with tqdm(range(n_steps), unit='steps') as tstep:
        for step in tstep:
            tstep.set_description(f'Step {step+1}')
            optimization_step()
            step_elbo = -training_loss().numpy()
            elbos.append(step_elbo)
            tstep.set_postfix(ELBO=step_elbo)

            if check_conv:
                if (step+1) >= 2*window and (step+1) % window == 0:
                    prev_avg = np.mean(elbos[-2*window:-window])
                    curr_avg = np.mean(elbos[-window:])
                    p_change = np.abs((curr_avg - prev_avg) / prev_avg)

                    if p_change <= tol:
                        patience_cnt += 1
                        print(f'Remaining patience: {patience - patience_cnt}')

                    if patience_cnt >= patience:
                        print('Model convergence. Stopping training.')
                        break

            if check_budget:
                current_time = datetime.now()

                if current_time >= start_time + time_budget:
                    print('Time budget reached. Stopping training.')
                    break

    return elbos

def train_lmc_svgp(
    train_params, 
    train_times, 
    Y_train, 
    random_init=False,
    seed=0, 
    natgrad=True, 
    verbose=False,
    check_conv=False,
    window=40,
    tol=1e-5,
    patience=5,
    check_budget=False,
    start_time=None,
    time_budget=None
):
    """Trains an LMC-SVGP model."""

    T_train = train_times.size # Number of training data points
    P = Y_train.shape[0] # Number of outputs
    M = train_params['M'] # Number of inducing points
    kernel_type = train_params['kernel_type'] # Type of kernel function
    batch_size = train_params['batch_size'] # Minibatch size
    n_steps = train_params['n_steps'] # Number of training steps
    lr = train_params['lr'] # Learning rate for ADAM
    gamma = train_params['gamma'] # Learning rate for NGD
    fix_ind = train_params['fix_ind'] # Fix inducing points
    ind_init_mode = train_params['ind_init_mode']
    ind_times = train_params['ind_times']

    if 'margin' in train_params.keys():
        margin = train_params['margin']
    else:
        margin = 0.15

    # Model
    lmc_params = dict(M=M, P=P, L=train_params['L'], kernel_type=kernel_type, ind_init_mode=ind_init_mode)
    model = get_lmc_svgp(lmc_params, train_times, fix_ind=fix_ind, random_init=random_init, seed=seed,
                         verbose=verbose, margin=margin, ind_times=ind_times)

    # Dataset
    #train_dataset = tf.data.Dataset.from_tensor_slices((train_times, Y_train.T)).repeat().shuffle(T_train)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_times, Y_train.T))
    train_dataset = train_dataset.shuffle(T_train, seed=seed).repeat()

    # Optimize
    start_time = datetime.now()

    if natgrad:
        elbos = run_adam_natgrad(model, 
                                 train_dataset, 
                                 T_train, 
                                 batch_size, 
                                 n_steps, 
                                 lr=lr, 
                                 gamma=gamma, 
                                 check_conv=check_conv, 
                                 window=window,
                                 tol=tol, 
                                 patience=patience,
                                 check_budget=check_budget,
                                 start_time=datetime.now(),
                                 time_budget=time_budget)
    else:
        elbos = run_adam(model, 
                         train_dataset, 
                         T_train, 
                         batch_size, 
                         n_steps,
                         lr=lr, 
                         check_conv=check_conv, 
                         window=window, 
                         tol=tol,
                         patience=patience,
                         check_budget=check_budget,
                         start_time=datetime.now(),
                         time_budget=time_budget)

    train_time = datetime.now() - start_time
    print(f'[Total training time] {str(train_time).split(".")[0]}')

    return model, elbos

def lmc_svgp_predict(model, data):
    """Performs approximate posterior inference with LMC on given batch of data."""

    times, Y = data
    P, T = Y.shape
    var = model.likelihood.variance.numpy()

    F_mean, F_var = model.predict_f(times)
    _, F_cov = model.predict_f(times, full_output_cov=True) # [T,P,P]
    F_cov = np.asarray(F_cov)
    F_mean = np.transpose(F_mean) # [P,T]
    F_var = np.transpose(F_var) # [P,T]

    Y_mean, Y_var = model.predict_y(times)
    Y_mean = np.transpose(Y_mean) # [P,T]
    Y_var = np.transpose(Y_var) # [P,T]
    Y_cov = F_cov + var * np.eye(P)
    Y_cov = np.asarray(Y_cov) # Convert to np.array

    preds = dict(F_mean=F_mean, F_var=F_var, F_cov=F_cov, 
                 Y_mean=Y_mean, Y_var=Y_var, Y_cov=Y_cov)

    return preds

def compute_metrics_lmc_svgp(model, data, preds):
    """Computes accuracy metrics on predictions."""

    _, Y = data
    P, T = Y.shape
    F_mean = preds['F_mean'] # [P,T]
    F_var = preds['F_var'] # [P,T]
    Y_mean = preds['Y_mean'] # [P,T]
    Y_cov = preds['Y_cov'] # [T,P,P]

    # MAE
    mae = np.mean(np.abs(Y - Y_mean))

    # MSE
    mse = np.mean((Y - Y_mean)**2)

    # Negative log-predictive density (NLPD)
    nlpd = NLPD(Y, Y_mean, Y_cov)
    
    # Collect metrics
    metrics = dict(MAE=mae, MSE=mse, NLPD=nlpd)

    return metrics
