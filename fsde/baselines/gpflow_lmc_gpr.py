import sys
sys.path.append('../../')

import os
import os.path as osp
import typing
from typing import Any, Callable, Optional, Union
from datetime import datetime
from tqdm import tqdm
import scipy
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

import gpflow
from gpflow.models import SVGP, GPModel
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor
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
from gpflow.logdensities import multivariate_normal
from gpflow.utilities import (
    print_summary, 
    set_trainable, 
    to_default_float, 
    triangular, 
    positive
)
from gpflow.base import Parameter

JITTER = default_jitter()

def vec(A):
    """Vectorizes a given matrix."""

    return tf.reshape(tf.transpose(A), (-1,1)) # [P,T] -> [PT,1]

def vec_inv(A, shape):
    """Unvectorizes a given vector."""

    p, n = shape

    return tf.transpose(tf.reshape(A, (n,p))) 

def quadratic(A, B, jitter=1e-8):
    """Computes A^T B^{-1} A, assuming B is PD."""

    d = B.shape[-1]
    L = tf.linalg.cholesky(B + jitter * tf.eye(d, dtype=tf.float64))
    S = tf.linalg.triangular_solve(L, A, lower=True)

    return tf.transpose(S) @ S

def inv_product(A, B, jitter=1e-8):
    """Computes B^{-1} A, assuming B is PD."""

    d = B.shape[-1]
    L = tf.linalg.cholesky(B + jitter * tf.eye(d, dtype=tf.float64))
    S = tf.linalg.triangular_solve(L, A, lower=True)
    S2 = tf.linalg.triangular_solve(tf.transpose(L), S, lower=False)

    return S2

def assert_params_false(
    called_method: Callable[..., Any],
    **kwargs: bool,
) -> None:
    """
    Asserts that parameters are ``False``.
    :param called_method: The method or function that is calling this. Used for nice error messages.
    :param kwargs: Parameters that must be ``False``.
    :raises NotImplementedError: If any ``kwargs`` are ``True``.
    """
    errors_str = ", ".join(f"{param}={value}" for param, value in kwargs.items() if value)
    if errors_str:
        raise NotImplementedError(
            f"{called_method.__qualname__} does not currently support: {errors_str}"
        )

class LMC_GPR(GPModel, InternalDataTrainingLossMixin):
    """Exact inference LMC, adapted from GPflow GPR implementation."""

    #@check_shapes(
    #    "data[0]: [N, D]", # Input
    #    "data[1]: [N, P]", # Output
    #    "noise_variance: []",
    #)
    def __init__(
        self,
        data,
        kernel,
        mean_function=None,
        noise_variance=None,
        likelihood=None,
    ):
        assert (noise_variance is None) or (
            likelihood is None
        ), "Cannot set both `noise_variance` and `likelihood`."

        if likelihood is None:
            if noise_variance is None:
                noise_variance = 1.0
            likelihood = gpflow.likelihoods.Gaussian(noise_variance)
        
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=kernel.W.shape[-1])
        self.data = data_input_to_tensor(data) # Store data as tf.Tensors

    # type-ignore is because of changed method signature:
    #@inherit_check_shapes
    def maximum_log_likelihood_objective(self) -> tf.Tensor:  # type: ignore[override]
        return self.log_marginal_likelihood()

    #@check_shapes(
    #    "return: []",
    #)
    def log_marginal_likelihood(self) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood.
        .. math::
            \log p(Y | \theta).
        """
        X, Y = self.data # [N,D], [N,P]
        N, P = Y.shape
        K = self.kernel(X, X, full_cov=True) # [N,P,N,P] -> Need to flatten to [NP,NP]
        K = tf.reshape(K, (N*P, N*P)) # N blocks of size P on each dimension
        #ks = add_likelihood_noise_cov(K, self.likelihood, X)
        diag_part = tf.linalg.diag_part(K)
        K = tf.linalg.set_diag(K, diag_part + self.likelihood.variance)
        #L = tf.linalg.cholesky(ks)
        #L = tf.linalg.cholesky(K + default_jitter() * tf.eye(N*P))
        L = tf.linalg.cholesky(K + JITTER * tf.eye(N*P, dtype=tf.float64)) # [NP,NP]
        #m = self.mean_function(X) # if None, this is a zerot function
        #m = tf.zeros_like(vec(tf.transpose(Y))) # [NP,1]
        m = tf.zeros((N*P,1), dtype=tf.float64)

        # vec(Y): [NP,1] # [P stacks of N-dim vectors]
        # vec(tf.transpose(Y)): [P,N] -> [NP,1]
        log_prob = multivariate_normal(vec(tf.transpose(Y)), m, L)
        
        return tf.reduce_sum(log_prob)

    #@inherit_check_shapes
    def predict_f(
        self, 
        Xnew, 
        full_cov=False, 
        full_output_cov=False # [N,P,P]
    ):
        r"""
        This method computes predictions at X \in R^{N \x D} input points
        .. math::
            p(F* | Y)
        where F* are points on the GP at new data points, Y are noisy observations at training data
        points.
        """
        assert_params_false(self.predict_f, full_cov=full_cov)

        X, Y = self.data
        M, P = Y.shape
        N, _ = Xnew.shape
        #err = Y - self.mean_function(X) # [N,1]
        err = vec(tf.transpose(Y)) # [MP,1

        kmm = self.kernel(X, X, full_cov=True) # [M,P,M,P]
        kmm = tf.reshape(kmm, (M*P, M*P)) # MxM blocks of size PxP
        diag_part = tf.linalg.diag_part(kmm)
        kmm = tf.linalg.set_diag(kmm, diag_part + self.likelihood.variance)

        knn = self.kernel(Xnew, Xnew, full_cov=True) # [N,P,N,P]
        knn = tf.reshape(knn, (N*P, N*P)) # NxN blocks of size PxP

        kmn = self.kernel(X, Xnew, full_cov=True) # [M,P,N,P]
        kmn = tf.reshape(kmn, (M*P, N*P)) # MxN blocks of size PxP

        #conditional = gpflow.conditionals.base_conditional
        #f_mean_zero, f_var = conditional(
        #    kmn, kmm, knn, err, full_cov=full_cov, white=False
        #)  # [N, P], [N, P] or [P, N, N]
        #f_mean = f_mean_zero #+ self.mean_function(Xnew)
        
        L = tf.linalg.cholesky(kmm + JITTER * tf.eye(M*P, dtype=tf.float64))
        S = tf.linalg.triangular_solve(L, kmn, lower=True)
        S2 = tf.linalg.triangular_solve(tf.transpose(L), S, lower=False)

        f_mean = tf.transpose(S2) @ err # [NP,1]
        f_mean = tf.transpose(vec_inv(f_mean, (P,N))) # [P,N] -> [N,P]

        if full_output_cov:
            f_full_cov = knn - tf.transpose(S) @ S # [NP,NP]
            f_cov = np.zeros((N,P,P), dtype=np.float64)

            for i in range(N):
                f_cov[i] = f_full_cov[i*P:(i+1)*P, i*P:(i+1)*P].numpy()

            return f_mean, f_cov

        else:
            f_var = tf.linalg.diag_part(knn) - \
                    tf.linalg.einsum('ij,ji->i', tf.transpose(S), S) # [NP,]
            f_var = tf.transpose(vec_inv(f_var, (P,N))) # [P,N] -> [N,P] 

            return f_mean, f_var

    #@inherit_check_shapes
    def predict_y(
        self, 
        Xnew, 
        full_cov=False, 
        full_output_cov=False
    ):
        assert_params_false(self.predict_y, full_cov=full_cov)

        if full_output_cov:
            f_mean, f_cov = self.predict_f(Xnew, full_output_cov=True)
            y_mean = f_mean # [N,P]
            N, P = f_cov.shape[:-1]
            eyes = tf.repeat(tf.eye(P, dtype=tf.float64)[None,:], N, axis=0)
            y_cov = f_cov + self.likelihood.variance * eyes

            return y_mean, y_cov

        else:
            f_mean, f_var = self.predict_f(Xnew, full_output_cov=False)
            y_mean = f_mean # [N,P]
            y_var = f_var + self.likelihood.variance * tf.ones_like(f_var, dtype=tf.float64)

            return y_mean, y_var

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
        nlpd += 0.5 * tf.squeeze(quadratic(residual, Y_cov_i, jitter=JITTER))

        return nlpd

    zipped = list(zip(Y.T, Y_mean.T, Y_cov))
    nlpds = list(map(_nlpd, zipped))

    return (1./(P*T)) * np.sum(nlpds)

def get_lmc_gpr(
    lmc_gpr_params, 
    train_times, 
    Y_train,
    seed=0,
    verbose=False
):
    """Returns a GPR model for LMC."""

    kernel_type = lmc_gpr_params['kernel_type']
    lengthscale = lmc_gpr_params['lengthscale']
    kernel_var = lmc_gpr_params['kernel_var']
    L = lmc_gpr_params['L']
    P, T = Y_train.shape

    if kernel_type == 'SE':
        kernel_list = [SquaredExponential() for _ in range(L)]

    elif kernel_type == 'Matern32':
        kernel_list = [Matern32() for _ in range(L)]

    elif kernel_type == 'Matern52':
        kernel_list = [Matern52() for _ in range(L)]

    np.random.seed(seed)
    W = np.random.randn(P,L)
    kernel = LinearCoregionalization(kernel_list, W=W)

    model = LMC_GPR(data=(train_times, Y_train.T), kernel=kernel)

    if verbose:
        print_summary(model)

    return model

def train_lmc_gpr(train_params, train_times, Y_train, verbose=False):
    """Trains a LMC-GPR model with the marginal log-likelihood."""

    T_train = train_times.size
    P = Y_train.shape[0] # Number of outputs
    kernel_type = train_params['kernel_type']
    n_steps = train_params['n_steps']
    lengthscale = train_params['lengthscale']
    kernel_var = train_params['kernel_var']
    L = train_params['L']

    # Model
    lmc_gpr_params = dict(kernel_type=train_params['kernel_type'],
                          lengthscale=train_params['lengthscale'],
                          kernel_var=train_params['kernel_var'],
                          L=train_params['L'])

    model = get_lmc_gpr(lmc_gpr_params, train_times, Y_train, verbose=verbose)

    # Optimize with L-BFGS
    start_time = datetime.now()
    optimizer = gpflow.optimizers.Scipy()
    opt_log = optimizer.minimize(model.training_loss, 
                                 model.trainable_variables,
                                 options=dict(maxiter=n_steps))

    train_time = datetime.now() - start_time
    print(f'[Total training time] {str(train_time).split(".")[0]}')

    return model, opt_log

def lmc_gpr_predict(model, data):
    """Performs exact posterior inference with LMC-GPR on given batch of data."""

    times, Y = data
    P, T = Y.shape

    F_mean, F_var = model.predict_f(times)
    _, F_cov = model.predict_f(times, full_output_cov=True) # [T,P,P]
    F_mean = np.transpose(F_mean) # [P,T]
    F_var = np.transpose(F_var) # [P,T]

    Y_mean, Y_var = model.predict_y(times)
    _, Y_cov = model.predict_y(times, full_output_cov=True) # [T,P,P]
    Y_mean = np.transpose(Y_mean) # [P,T]
    Y_var = np.transpose(Y_var) # [P,T]

    preds = dict(F_mean=F_mean, F_var=F_var, F_cov=F_cov, 
                 Y_mean=Y_mean, Y_var=Y_var, Y_cov=Y_cov)

    return preds

def compute_metrics_lmc_gpr(model, data, preds):
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

