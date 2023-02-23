import sys
sys.path.append('../../') # Add FSDE root

import chex
import typing
from typing import Any, Callable, Optional, Union
import jaxtyping
from jaxtyping import f64, i64, PyTree

import jax
jax.config.update("jax_enable_x64", True) # For setting default dtype to float64
import jax.numpy as jnp
import jax.random as jr
from jax.nn import softplus
from jax import grad, jit, vmap, pmap, tree_map, lax, value_and_grad, custom_vjp, jvp, vjp, tree_util
from jax.tree_util import Partial
import jax.scipy as jscipy
from jax.scipy.linalg import expm
from jax.experimental.ode import odeint
from jax.experimental import host_callback
from jax.example_libraries.optimizers import clip_grads
import os
from functools import partial
from tqdm import tqdm
from datetime import datetime
import time
from time import sleep
import numpy as np
import copy
import scipy
from scipy.linalg import solve_continuous_lyapunov
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans 

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

import optax

import fsde
from fsde.core.ops import *

EPS = jnp.float64(1e-3)

def loglinear_schedule(init_value, end_value, transition_steps, transition_begin=0):
    """Log-linearly "increasing" learning rate scheduler for optax optimizer."""

    def schedule(count):
        count = jnp.clip(count - transition_begin, 0, transition_steps)
        factor = (jnp.log(end_value) - jnp.log(init_value)) / transition_steps

        return init_value * jnp.exp(count * factor)

    return schedule

def kfold_split(times, F, Y, n_ext, n_splits=5, shuffle=True):
    """Generates a K-fold split dataset."""

    T = times.size
    idx = np.arange(T)
    data = {}
    
    # Include full data
    full_data = dict(Y=Y, F=F, times=times)
    data['full'] = full_data
    
    # Hold out extrapolation data
    idx = np.arange(T)
    ext_idx = idx[-n_ext:]
    ext_data = dict(Y=Y[:,ext_idx], 
                    F=F[:,ext_idx], 
                    times=times[ext_idx],
                    idx=ext_idx)
    data['test_ext'] = ext_data
    
    # K-fold train-interpolation split
    rem_idx = np.arange(T-n_ext)
    kfold = KFold(n_splits=n_splits, shuffle=shuffle)
    kfold.get_n_splits(rem_idx)
    
    for split_id, (train_idx, int_idx) in enumerate(kfold.split(rem_idx)):
        # Train data for split
        train_split_times = times[train_idx]
        Y_train_split = Y[:,train_idx]
        F_train_split = F[:,train_idx]
        train_split_data = dict(Y=Y_train_split,
                                F=F_train_split,
                                times=train_split_times,
                                idx=train_idx)
        
        # Interpolation data for split
        int_split_times = times[int_idx]
        Y_int_split = Y[:,int_idx]
        F_int_split = F[:,int_idx]
        int_split_data = dict(Y=Y_int_split,
                              F=F_int_split,
                              times=int_split_times,
                              idx=int_idx)
        
        split = dict(train=train_split_data,
                     test_int=int_split_data)
        
        data[f'split_{split_id+1}'] = split
        
    return data

@chex.dataclass
class Dataset:
    """Dataset class adapted from GPJax (Pinder and Dodd, 2022)"""
    
    times: chex.ArrayDevice # [T,]
    Y: chex.ArrayDevice # [P,T]
        
    def __repr__(self):
        return f'Time Points: {self.times.shape}, Outputs: {self.Y.shape}'
    
    def __add__(self, other: "Dataset") -> "Dataset":
        times = jnp.concatenate((self.times, other.times))
        Y = jnp.concatenate((self.Y, other.Y), axis=1)
        
        return Dataset(times=times, Y=Y)
    
    @property
    def T(self) -> int: 
        """Number of time points in the dataset"""
        return self.times.size
    
    @property
    def P(self) -> int:
        """Number of outputs in the dataset"""
        return self.Y.shape[0]

def get_batch(dataset: Dataset, batch_size: int, key: chex.PRNGKey):
    """Mini-batching function adapted from GPJax"""
    
    # Randomly sample mini-batch indices
    idxs = jr.choice(key, dataset.T, (batch_size,), replace=False) # TESTING replacement
    
    return Dataset(times=dataset.times[idxs], Y=dataset.Y[:,idxs])

def progress_bar_scan(n_iters: int, log_rate: int):
    """Progress bar for Jax.lax scans (adapted from https://www.jeremiecoullon.com/2021/01/29/jax_progress_bar/)."""

    tqdm_bars = {}
    remainder = n_iters % log_rate

    def _define_tqdm(args, transform):
        tqdm_bars[0] = tqdm(range(n_iters))

    def _update_tqdm(args, transform):
        loss_val, arg = args
        tqdm_bars[0].update(arg)
        tqdm_bars[0].set_postfix({"ELBO": f"{loss_val: .2f}"})

    def _update_progress_bar(loss_val, i):
        """Updates tqdm progress bar of a JAX scan or loop."""
        _ = lax.cond(
            i == 0,
            lambda _: host_callback.id_tap(_define_tqdm, None, result=i),
            lambda _: i,
            operand=None,
        )

        _ = lax.cond(
            # update tqdm every multiple of `print_rate` except at the end
            (i % log_rate == 0) & (i != n_iters - remainder),
            lambda _: host_callback.id_tap(
                _update_tqdm, (loss_val, log_rate), result=i
            ),
            lambda _: i,
            operand=None,
        )

        _ = lax.cond(
            # update tqdm by `remainder`
            i == n_iters - remainder,
            lambda _: host_callback.id_tap(
                _update_tqdm, (loss_val, remainder), result=i
            ),
            lambda _: i,
            operand=None,
        )

    def _close_tqdm(args, transform):
        tqdm_bars[0].close()

    def close_tqdm(result, i):
        return lax.cond(
            i == n_iters - 1,
            lambda _: host_callback.id_tap(_close_tqdm, None, result=result),
            lambda _: result,
            operand=None,
        )

    def _progress_bar_scan(func):
        """Decorator that adds a progress bar to `body_fun` used in `lax.scan`."""

        def wrapper_progress_bar(carry, x):
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x
            result = func(carry, x)
            *_, loss_val = result
            _update_progress_bar(loss_val, iter_num)
            return close_tqdm(result, iter_num)

        return wrapper_progress_bar

    return _progress_bar_scan

def progress_bar_scan2(n_iters: int, log_rate: int):
    """Progress bar for Jax.lax scans (adapted from https://www.jeremiecoullon.com/2021/01/29/jax_progress_bar/)."""

    tqdm_bars = {}
    remainder = n_iters % log_rate

    def _define_tqdm(args, transform):
        tqdm_bars[0] = tqdm(range(n_iters))

    def _update_tqdm(args, transform):
        loss_val, exp_loglik, kl, arg = args
        tqdm_bars[0].update(arg)
        tqdm_bars[0].set_postfix({"ELBO": f"{loss_val: .2f}",
                                  "exp_loglik": f"{exp_loglik: .2f}",
                                  "KL": f"{kl: .2f}"})

    def _update_progress_bar(loss_val, exp_loglik, kl, i):
        """Updates tqdm progress bar of a JAX scan or loop."""
        _ = lax.cond(
            i == 0,
            lambda _: host_callback.id_tap(_define_tqdm, None, result=i),
            lambda _: i,
            operand=None,
        )

        _ = lax.cond(
            # update tqdm every multiple of `print_rate` except at the end
            (i % log_rate == 0) & (i != n_iters - remainder),
            lambda _: host_callback.id_tap(
                _update_tqdm, (loss_val, exp_loglik, kl, log_rate), result=i
            ),
            lambda _: i,
            operand=None,
        )

        _ = lax.cond(
            # update tqdm by `remainder`
            i == n_iters - remainder,
            lambda _: host_callback.id_tap(
                _update_tqdm, (loss_val, exp_loglik, kl, remainder), result=i
            ),
            lambda _: i,
            operand=None,
        )

    def _close_tqdm(args, transform):
        tqdm_bars[0].close()

    def close_tqdm(result, i):
        return lax.cond(
            i == n_iters - 1,
            lambda _: host_callback.id_tap(_close_tqdm, None, result=result),
            lambda _: result,
            operand=None,
        )

    def _progress_bar_scan(func):
        """Decorator that adds a progress bar to `body_fun` used in `lax.scan`."""

        def wrapper_progress_bar(carry, x):
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x
            result = func(carry, x)
            *_, metrics = result
            _update_progress_bar(*metrics, iter_num)
            return close_tqdm(result, iter_num)

        return wrapper_progress_bar

    return _progress_bar_scan

def gh_quadrature(func, mean, var, n_gh=20):
    """
    Computes the Gaussian-Hermite quadrature for func, shifted and scaled by mean and var.

    func: log-probabilty density function
    mean: Mean of the distribution
    var: Variance of the distribution
    n_gh: Number of quadrature points
    """
    mean = mean.squeeze()
    var = var.squeeze()

    gh_points, gh_weights = np.polynomial.hermite.hermgauss(n_gh)
    std = jnp.sqrt(var)
    adjusted_gh_points = mean + jnp.sqrt(2.) * std * gh_points # [n_gh,]
    adjusted_gh_weights = gh_weights / jnp.sqrt(jnp.pi)

    return jnp.sum(func(adjusted_gh_weights) * adjusted_gh_points)

def Bernoulli_loglik(y, p):
    """Log-likelihood function for Bernoulli distribution"""

    y = y.squeeze()
    p = p.squeeze()

    return y * jnp.log(p) + (1-y) * jnp.log(1-p)

def Poisson_loglik(y, rate):
    """Log-likelihood function for Poisson distribution"""

    y = y.squeeze()
    rate = rate.squeeze()

    return y * jnp.log(rate) - rate - (y * jnp.log(y) - y)

def probit_inv_link(x):
    """Probit inverse link function"""

    return jscipy.stats.norm.cdf(x)

def Poisson_inv_link(x):
    """Poisson inverse link function"""

    return jnp.exp(x)

def get_state_dim(kernel):
    """Returns the state dimension for the given kernel."""

    if kernel == 'Matern32':
        return 2
        
    elif kernel == 'Matern52':
        return 3

    elif kernel == 'SE':
        return 6

    else:
        raise RuntimeError('[Error] Invalid kernel specified.')

def Matern32_binom(lengthscale, nu=1.5):
    """Computes the binomial coefficients for Matern3/2 kernel."""

    coeff = jnp.sqrt(2 * nu) / lengthscale.squeeze()
    binom = jnp.array([-coeff**2, -2 * coeff])

    return coeff, binom

def Matern52_binom(lengthscale, nu=2.5):
    """Computes the binomial coefficients for Matern5/2 kernel."""

    coeff = jnp.sqrt(2 * nu) / lengthscale.squeeze()
    binom = jnp.array([-coeff**3, -3 * coeff**2, -3 * coeff])

    return coeff, binom

def compute_Q(k_var, coeff, nu=1.5):
    """Computes the spectral density of the white noise process."""

    p = nu - 0.5
    Q = 2 * k_var * jnp.sqrt(jnp.pi) * coeff**(2*p+1) * gamma(p+1) / gamma(p+0.5)
        
    return Q

def sde_params_Matern32(unc_k_var, unc_lengthscale):
    """Returns the SDE parameters for the Matern3/2 kernel."""

    k_var = softplus(unc_k_var, low=EPS)
    lengthscale = softplus(unc_lengthscale, low=EPS)
            
    # Compute Matern3/2 binomial coefficients
    coeff, binom = Matern32_binom(lengthscale, nu=1.5)
            
    # Build F
    F = jnp.eye(2, k=1)
    F = F.at[-1,:].set(binom) # [D,D]
            
    # Compute Q
    Q = compute_Q(k_var, coeff)
            
    return F, Q

def compute_F_Matern32(unc_lengthscale):
    """Computes the state-transition matrix for the Matern 3/2 kernel."""

    lengthscale = softplus(unc_lengthscale, low=EPS)

    # Compute Matern3/2 binomial coefficients
    _, binom = Matern32_binom(lengthscale, nu=1.5)

    # Build F
    F = jnp.eye(2, k=1)
    F = F.at[-1,:].set(binom) # [D,D]

    return F

def sde_params_Matern52(unc_k_var, unc_lengthscale):
    """Returns the SDE parameters for the Matern5/2 kernel."""

    k_var = softplus(unc_k_var, low=EPS)
    lengthscagle = softplus(unc_lengthscale, low=EPS)
    
    # Compute Matern5/2 binomial coefficients
    coeff, binom = Matern52_binom(lengthscale, nu=2.5)

    # Build F
    F = jnp.eye(3, k=1)
    F = F.at[-1,:].set(binom) # [D,D]

    # Compute Q
    Q = compute_Q(k_var, coeff)

    return F, Q

def compute_F_Matern52(unc_lengthscale):
    """Computes the state-transition matrix for the Matern 5/2 kernel."""

    lengthscale = softplus(unc_lengthscale, low=EPS)

    # Compute Matern5/2 binomial coefficients
    _, binom = Matern52_binom(lengthscale, nu=2.5)

    # Build F
    F = jnp.eye(3, k=1)
    F = F.at[-1,:].set(binom) # [D,D]

    return F
    
def cov_infty_Matern32(unc_k_vars, unc_lengthscales):

    def helper(unc_k_var, unc_lengthscale):
        k_var = softplus(unc_k_var, low=EPS).squeeze()
        lengthscale = softplus(unc_lengthscale, low=EPS).squeeze()
        coeff, _ = Matern32_binom(lengthscale)
        diag_vals = jnp.array([k_var, k_var * coeff**2])
        cov_infty_i = jnp.diag(diag_vals)

        return cov_infty_i

    cov_infty = vmap(helper, in_axes=(0,0))(unc_k_vars, unc_lengthscales)

    return cov_infty

def cov_infty_Matern52(unc_k_vars, unc_lengthscales):
    
    def helper(unc_k_var, unc_lengthscale):
        k_var = softplus(unc_k_var, low=EPS).squeeze()
        lengthscale = softplus(unc_lengthscale, low=EPS).squeeze()
        coeff, _ = Matern52_binom(lengthscale)
        cov_infty_i = jnp.array([[k_var, 0., -(1/3.) * k_var * coeff**2],
                                 [0., (1/3.) * k_var * coeff**2, 0.],
                                 [-(1/3.) * k_var * coeff**2, 0., k_var * coeff**4]])

        return cov_infty_i

    cov_infty = vmap(helper, in_axes=(0,0))(unc_k_vars, unc_lengthscales)

    return cov_infty

def compute_psi(interval, F_i):
    """Computes the state-transition matrix."""

    return expm(interval * F_i)

def cov_steady(interval, F_i, cov_infty_i):
    """Computes transition covariance assuming steady-state conditions."""

    psi_i_k = compute_psi(interval, F_i)
        
    return cov_infty_i - psi_i_k @ cov_infty_i @ psi_i_k.T

def constrain_R(
    unc_R_i_diag_blocks: f64['M D*(D+1)/2'], 
    unc_R_i_offdiag_blocks: f64['M-1 D D']
) -> tuple[f64['M D D'], f64['M-1 D D']]:
    """Constrains the tridiagonal blocks of R."""

    triangular = tfb.FillTriangular(upper=False)
    softplus_diag = tfb.TransformDiagonal(diag_bijector=tfb.Softplus(low=EPS))
    R_i_diag_blocks = triangular.forward(unc_R_i_diag_blocks) # Make triangular [M,D,D]
    R_i_diag_blocks = softplus_diag.forward(R_i_diag_blocks) # Softplus diagonal [M,D,D]
    R_i_diag_blocks = triangular.inverse(R_i_diag_blocks) # Make compact [M,D*(D+1)/2]
    R_i_offdiag_blocks = unc_R_i_offdiag_blocks # [M-1,D,D]

    return R_i_diag_blocks, R_i_offdiag_blocks

def unconstrain_R(
    R_i_diag_blocks: f64['M D D'], 
    R_i_offdiag_blocks: f64['M-1 D D']
) -> tuple[f64['M D*(D+1)/2'], f64['M-1 D D']]:
    """Unconstrains the tridiagonal blocks of R."""

    triangular = tfb.FillTriangular(upper=False)
    softplus_diag = tfb.TransformDiagonal(diag_bijector=tfb.Softplus(low=EPS))
    unc_R_i_diag_blocks = softplus_diag.inverse(R_i_diag_blocks) # [M,D,D]
    unc_R_i_diag_blocks = triangular.inverse(unc_R_i_diag_blocks) # [M,D*(D+1)/2]
    unc_R_i_offdiag_blocks = R_i_offdiag_blocks # [M-1,D,D]

    return unc_R_i_diag_blocks, unc_R_i_offdiag_blocks

def init_R(
    L: int, 
    M: int, 
    D: int,
    random_init: Optional[bool] = False,
    seed: Optional[chex.PRNGKey] = jr.PRNGKey(0)
) -> tuple[f64['L M D*(D+1)/2'], f64['L M-1 D D']]:
    """Initializes the variational precision blocks to identity."""

    def _init():
        # Set up diagonal blocks
        unc_R_diag_blocks = inv_softplus(1.) * jnp.eye(D)[None,None,:,:] # [1,1,D,D]
        unc_R_diag_blocks = jnp.repeat(unc_R_diag_blocks, M, axis=1) # [1,M,D,D]
        unc_R_diag_blocks = jnp.repeat(unc_R_diag_blocks, L, axis=0) # [L,M,D,D]

        # Set up off-diagonal blocks
        unc_R_offdiag_blocks = jnp.zeros((L,M-1,D,D)) # [L,M-1,D,D]

        # Extract non-zero entries
        triangular = tfb.FillTriangular(upper=False)
        unc_R_diag_blocks = vmap(triangular.inverse)(unc_R_diag_blocks) # [L,M,D*(D+1)/2]

        return unc_R_diag_blocks, unc_R_offdiag_blocks

    def _random_init():
        normal = tfd.Normal(loc=0., scale=1.)
        n_triangular = int(D*(D+1) / 2)
        subkeys = jr.split(seed, num=L)
        unc_R_diag_blocks = vmap(normal.sample, in_axes=(None,0))((M,n_triangular), subkeys) # [L,M,D*(D+1)/2]
        unc_R_offdiag_blocks = vmap(normal.sample, in_axes=(None,0))((M-1,D,D), subkeys) # [L,M-1,D,D]
        unc_R_diag_blocks = unc_R_diag_blocks.astype(jnp.float64)
        unc_R_offdiag_blocks = unc_R_offdiag_blocks.astype(jnp.float64)

        return unc_R_diag_blocks, unc_R_offdiag_blocks

    unc_R_diag_blocks, unc_R_offdiag_blocks = lax.cond(random_init, _random_init, _init)

    return unc_R_diag_blocks, unc_R_offdiag_blocks

def init_params(
    kernel: str, # Kernel type
    L: int, # Number of latent GPs
    P: int, # Number of outputs
    M: int, # Number of (inducing) points
    lengthscale: Optional[f64] = 1.,
    k_var: Optional[f64] = 1.,
    var: Optional[f64] = 1.,
    random_init: Optional[bool] = False,
    key: Optional[chex.PRNGKey] = jr.PRNGKey(0)
) -> tuple[dict, dict, Callable, Callable]:
    """Initializes the parameters."""

    subkeys = jr.split(key, num=3)
    D = get_state_dim(kernel)

    if kernel == 'Matern32':
        compute_cov_infty = cov_infty_Matern32
        compute_F = compute_F_Matern32

    elif kernel == 'Matern52':
        compute_cov_infty = cov_infty_Matern52
        compute_F = compute_F_Matern52

    W = jr.normal(subkeys[0], (P,L)) # [P,L]
    unc_lengthscales = inv_softplus(lengthscale) * jnp.ones((L,1))
    unc_k_vars = inv_softplus(k_var) * jnp.ones((L,1))
    unc_var = inv_softplus(var)

    v_mean = jr.normal(subkeys[1], (L,D,M))
    unc_R_diag_blocks, unc_R_offdiag_blocks = init_R(L, M, D, random_init=random_init,
                                                     seed=subkeys[2])

    model_params = dict(W=W,
                        unc_lengthscales=unc_lengthscales,
                        unc_k_vars=unc_k_vars,
                        unc_var=unc_var)

    v_params = dict(v_mean=v_mean,
                    unc_R_diag_blocks=unc_R_diag_blocks,
                    unc_R_offdiag_blocks=unc_R_offdiag_blocks)

    return (model_params, v_params, compute_cov_infty, compute_F)

def set_ind_times(
    M: int,
    train_times: Union[f64['T 1'], f64['T']],
    mode: Optional[str] = 'equal',
    margin: Optional[f64] = 0.15
) -> f64['M']:
    """Determines the inducing time points via K-means or equal spacing."""

    if mode == 'K-means':
        if train_times.ndim == 1:
            train_times = train_times[:,None]

        kmeans = KMeans(n_clusters=M).fit(train_times) # Note: train_times must be in shape [T,1]
        ind_times = jnp.sort(kmeans.cluster_centers_.squeeze()) # [M,]

    elif mode == 'equal':
        train_times = train_times.squeeze() # [T,]
        first = train_times[0] - margin
        last = train_times[-1] + margin
        ind_times = jnp.linspace(first, last, M) # first, last inclusive

    return ind_times

def FSDE_NLPD(
    Y: f64['P T'],
    Y_hat: f64['P T'],
    Y_cov: f64['T P P'],
    jitter: Optional[f64] = 1e-10
) -> f64:
    """Computes the NLPD for predictions made by FSDE."""

    P, T = Y.shape

    @partial(vmap, in_axes=(1,1,0))
    def _nlpd(Y_i, Y_hat_i, Y_cov_i):

        logdet_cov_i = jnp.multiply(*jnp.linalg.slogdet(Y_cov_i))
        residual = (Y_i - Y_hat_i)[:,None] # [P,1]

        nlpd = 0.5 * P * jnp.log(2 * jnp.pi)
        nlpd += 0.5 * logdet_cov_i
        nlpd += 0.5 * quadratic(residual, Y_cov_i, jitter=jitter)

        return nlpd

    nlpd = (1./(P*T)) * jnp.sum(_nlpd(Y, Y_hat, Y_cov))

    return nlpd
