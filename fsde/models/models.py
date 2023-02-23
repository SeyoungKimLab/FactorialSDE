import sys
sys.path.append('../../') # Add FSDE root

import chex
import typing
from typing import Any, Callable, Optional, Union, Dict
import jaxtyping
from jaxtyping import f64, i64, PyTree

import jax
jax.config.update("jax_enable_x64", True) # For setting default dtype to float64
import jax.numpy as jnp
import jax.random as jr
from jax.nn import softplus
from jax import grad, jit, vmap, pmap, tree_map, lax, value_and_grad, custom_vjp, jvp, vjp
from jax.tree_util import Partial
import jax.scipy as jscipy
from jax.scipy.linalg import expm
from jax.experimental.ode import odeint
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
from fsde.core.model_utils import *

EPS = jnp.float64(1e-3)

@chex.dataclass
class FSDE:
    """Factorial SDE with Structured Mean-Field Variational Inference (FSDE)"""

    model_params: Dict
    v_params: Dict
    kernel: Dict
    train_dataset: "Dataset"
    compute_cov_infty: Callable
    compute_F: Callable
    jitter: f64

    @property
    def L(self) -> int:
        return self.model_params['W'].shape[1]

    @property
    def P(self) -> int:
        return self.model_params['W'].shape[0]

    @property
    def T(self) -> int:
        return self.train_dataset.T

    @property
    def D(self) -> int:
        return self.v_params['v_mean'].shape[1]

    @property
    def first_train_time(self) -> f64:
        return self.train_dataset.times[0]

    @property
    def last_train_time(self) -> f64:
        return self.train_dataset.times[-1]

    @property
    def params(self) -> Dict:
        return (self.model_params, self.v_params)

    def get_MOGP_params(self) -> Dict:
        """Returns the MOGP parameter values."""

        return self.model_params

    def get_v_params(self) -> Dict:
        """Returns the variational parameter values."""

        return self.v_params

    def precompute_pred_args(self):
        """Precomputes the arguments required for prediction."""

        # Compute state-transition matrix
        F = jit(vmap(self.compute_F))(self.model_params['unc_lengthscales'])

        # Compute steady-state covariance
        cov_infty = self.compute_cov_infty(self.model_params['unc_k_vars'], 
                                           self.model_params['unc_lengthscales']) # [L,D,D]

        # Constrain R
        R_diag_blocks, R_offdiag_blocks = jit(vmap(constrain_R, in_axes=(0,0)))(self.v_params['unc_R_diag_blocks'],
                                                                                self.v_params['unc_R_offdiag_blocks'])

        # Compute S from R
        # Note: S_diag_blocks [L,T,D,D], S_offdiag_blocks [L,T-1,D,D]
        S_diag_blocks, S_offdiag_blocks = jit(vmap(self.R_to_S, in_axes=(0,0)))(R_diag_blocks, 
                                                                                R_offdiag_blocks)
        
        # Compute inverse blocks of S
        # Note: S_inv_diag_blocks [L,T,D,D], S_offdiag_blocks [L,T-1,D,D]
        S_inv_diag_blocks, S_inv_offdiag_blocks = jit(vmap(self.tridiag_inv, in_axes=(0,0)))(S_diag_blocks,
                                                                                             S_offdiag_blocks)

        return (F, self.model_params['unc_var'], self.model_params['W'], cov_infty, 
                self.v_params['v_mean'], S_inv_diag_blocks, S_inv_offdiag_blocks)
    
    def R_to_S(
        self, 
        R_i_diag_blocks: f64['T D*(D+1)/2'], 
        R_i_offdiag_blocks: f64['T-1 D D']
    ) -> tuple[f64['T D D'], f64['T-1 D D']]:
        """
        Computes the variational precision matrix S from its constrained Cholesky factor R.
        
        Input shapes: R_i_diag_blocks [T,D*(D+1)/2], R_i_offdiag_blocks [T-1,D,D]
        Output shapes: S_i_diag_blocks [T,D,D], S_i_offdiag_blocks [T-1,D,D]       
        """

        # Convert diagonal blocks to lower triangular blocks
        triangular = tfb.FillTriangular(upper=False)
        R_i_diag_blocks = triangular.forward(R_i_diag_blocks) # [T,D,D]

        # Compute the diagonal blocks
        S_i_diag_blocks_1 = jnp.einsum('ijk,ilk->ijl', R_i_diag_blocks, R_i_diag_blocks) # [T,D,D]
        S_i_diag_blocks_2 = jnp.einsum('ijk,ilk->ijl', R_i_offdiag_blocks, R_i_offdiag_blocks) # [T-1,D,D]
        S_i_diag_blocks = S_i_diag_blocks_1 # [T,D,D]
        S_i_diag_blocks = S_i_diag_blocks.at[1:,:,:].add(S_i_diag_blocks_2) # [T,D,D]

        # Compute the off-diagonal blocks
        S_i_offdiag_blocks = jnp.einsum('ijk,ilk->ijl', R_i_offdiag_blocks, R_i_diag_blocks[:-1,:,:]) # [T-1,D,D]

        return S_i_diag_blocks, S_i_offdiag_blocks

    def S_to_R(
        self, 
        S_i_diag_blocks: f64['T D D'], 
        S_i_offdiag_blocks: f64['T-1 D D']
    ) -> tuple[f64['T D*(D+1)/2'], f64['T-1 D D']]:
        """
        Computes the unconstrained Cholesky factor R from the variational precision matrix S.
        
        Input shapes: S_i_diag_blocks [T,D,D], S_i_offdiag_blocks [T-1,D,D]
        Output shapes: unc_R_i_diag_blocks [T,D*(D+1)/2], unc_R_i_offdiag_blocks [T-1,D,D]
        """

        # Initial diagonal Cholesky block
        R_i_diag_block_1 = jnp.linalg.cholesky(S_i_diag_blocks[0,:,:]) # [D,D]

        # Computes \bmR_{\ell,i} and \bmR_{\ell,i,i-1} given \bmR_{\ell,i-1}
        def helper(R_i_diag_prev, idx):
            S_i_diag_block = lax.dynamic_slice(S_i_diag_blocks, (idx+1,0,0), (1,self.D,self.D)).squeeze()
            S_i_offdiag_block = lax.dynamic_slice(S_i_offdiag_blocks, (idx,0,0), (1,self.D,self.D)).squeeze()
            
            # Find \bmR_{\ell,i,i-1}
            R_i_offdiag_block = lax.linalg.triangular_solve(R_i_diag_prev.T, S_i_offdiag_block, left_side=False, lower=False)

            # Find \bmR_{\ell,i}
            res = S_i_diag_block - R_i_offdiag_block @ R_i_offdiag_block.T
            R_i_diag_block = jnp.linalg.cholesky(res)

            # Stack diagonal and off-diagonal blocks
            R_i_blocks = jnp.concatenate([R_i_diag_block[None,:,:], R_i_offdiag_block[None,:,:]]) # [2,D,D]

            return R_i_diag_block, R_i_blocks

        idxs = jnp.arange(0, self.T-1)
        _, R_i_blocks = lax.scan(helper, R_i_diag_block_1, idxs) # [M-1,2,D,D]

        # Diagonal blocks
        R_i_diag_blocks = R_i_blocks[:,0,:,:] # [M-1,D,D]
        R_i_diag_blocks = jnp.concatenate([R_i_diag_block_1[None,:,:],
                                           R_i_diag_blocks]) # [M,D,D]
        
        # Off-diagonal blocks
        R_i_offdiag_blocks = R_i_blocks[:,1,:,:] # [M-1,D,D]
        
        # Unconstrain R
        unc_R_i_diag_blocks, unc_R_i_offdiag_blocks = unconstrain_R(R_i_diag_blocks, R_i_offdiag_blocks)

        return unc_R_i_diag_blocks, unc_R_i_offdiag_blocks

    def solve_tridiag_system(
        self, 
        eta_i_1: f64['D T'], 
        S_i_diag_blocks: f64['T D D'], 
        S_i_offdiag_blocks: f64['T-1 D D']
    ) -> f64['D T']:
        """
        Solve the block-tridiagonal system of equations to recover the variational mean.

        Used in the nat_to_params() function.

        Input shapes: eta_i_1 [D,T], S_i_diag_blocks [T,D,D], S_i_offdiag_blocks [T-1,D,D]
        Output shapes: v_mean_i [D,T]
        """

        # STEP 1: Upper block-bidiagonalization
        eta_i_1_init = eta_i_1[:,0:1] # \bmeta_{\ell,1}^{(1)}, [D,1]
        S_i_diag_init = S_i_diag_blocks[0,:,:] # \bmS_{\ell,1}, [D,D]
        S_i_offdiag_init = S_i_offdiag_blocks[0,:,:] # \bmS_{\ell,21}, [D,D]
        
        off_init = inv_product(S_i_offdiag_init.T, S_i_diag_init, jitter=self.jitter) # [D,D]
        eta_init = inv_product(eta_i_1_init, S_i_diag_init, jitter=self.jitter) # [D,1]
        init = jnp.concatenate([off_init, eta_init], axis=-1) # [D,D+1]

        def bidiag_helper(prev, idx):
            # Unstack carry
            off_prev = prev[:,:-1] # [D,D]
            eta_prev = prev[:,-1:] # [D,1]

            eta_i_1_idx = lax.dynamic_slice(eta_i_1, (0,idx), (self.D,1)) # [D,1]
            S_i_diag_idx = lax.dynamic_slice(S_i_diag_blocks, (idx,0,0), (1,self.D,self.D)).squeeze() # [D,D]
            S_i_offdiag_idx = lax.dynamic_slice(S_i_offdiag_blocks, (idx-1,0,0), (1,self.D,self.D)).squeeze() # [D,D]
            S_i_offdiag_next = lax.dynamic_slice(S_i_offdiag_blocks, (idx,0,0), (1,self.D,self.D)).squeeze() # [D,D]

            # Compute \bmS_{\ell,i,i+1}^*
            res = S_i_diag_idx - S_i_offdiag_idx @ off_prev # \bmS_{\ell,i}^*, [D,D]
            off_new = inv_product(S_i_offdiag_next.T, res, jitter=self.jitter) # [D,D]

            # Compute \bmeta_{\ell,i}^{(1)**}
            res2 = eta_i_1_idx - S_i_offdiag_idx @ eta_prev # [D,1]
            eta_new = inv_product(res2, res, jitter=self.jitter) # [D,1]

            # Stack results
            stacked = jnp.concatenate([off_new, eta_new], axis=-1) # [D,D+1]

            return stacked, stacked

        idxs = jnp.arange(1, self.T-1)
        _, stacked = lax.scan(bidiag_helper, init, idxs) # [M-2,D,D+1]

        # Unstack result
        off_blocks = stacked[:,:,:-1] # [M-2,D,D]
        eta_blocks = stacked[:,:,-1:] # [M-2,D,1]

        # Compute last block
        eta_i_1_final = eta_i_1[:,-1:] # \bmeta_{\ell,M}^{(1)} # [D,1]
        S_i_diag_final = S_i_diag_blocks[-1,:,:] # \bmS_{\ell,M} # [D,D]
        S_i_offdiag_final = S_i_offdiag_blocks[-1,:,:] # \bmS_{\ell,M,M-1} # [D,D]
        off_final = S_i_diag_final - S_i_offdiag_final @ off_blocks[-1,:,:] # [D,D]
        eta_final = eta_i_1_final - S_i_offdiag_final @ eta_blocks[-1,:,:] # [D,1]

        off_blocks = jnp.concatenate([off_init[None,:,:], off_blocks, off_final[None,:,:]], axis=0) # [M,D,D]
        eta_blocks = jnp.concatenate([eta_init[None,:,:], eta_blocks, eta_final[None,:,:]], axis=0) # [M,D,1]

        # STEP 2: Back-substitution to find mean
        def back_sub_helper(v_mean_i_prev, idx):
            eta_block_idx = lax.dynamic_slice(eta_blocks, (idx,0,0), (1,self.D,1)).squeeze(axis=0) # [D,1]
            off_block_idx = lax.dynamic_slice(off_blocks, (idx,0,0), (1,self.D,self.D)).squeeze() # [D,D]

            v_mean_i_idx = (eta_block_idx - off_block_idx @ v_mean_i_prev) # [D,1]

            return v_mean_i_idx, v_mean_i_idx

        v_mean_i_final = inv_product(eta_blocks[-1,:,:], off_blocks[-1,:,:], jitter=self.jitter) # [D,1]
        idxs = jnp.flip(jnp.arange(0, self.T-1))
        _, v_mean_i = lax.scan(back_sub_helper, v_mean_i_final, idxs) # [M-1,D,1]
        v_mean_i = v_mean_i.squeeze(axis=-1) # [M-1,D]
        v_mean_i = jnp.flip(v_mean_i, axis=0).T # [D,M-1]
        v_mean_i = jnp.concatenate([v_mean_i, v_mean_i_final], axis=1) # [D,M]

        return v_mean_i

    def params_to_nat(
        self, 
        v_mean: f64['L D T'], 
        unc_R_diag_blocks: f64['L T D*(D+1)/2'], 
        unc_R_offdiag_blocks: f64['L T-1 D D']
    ) -> tuple[f64['L D T'], f64['L T D D'], f64['L T-1 D D']]:
        """
        Converts the unconstrained variational parameters to their corresponding natural parameters.

        Note: eta_1 = (RR^T)m, eta_2 = BTD(-0.5 * RR^T)

        Input shapes: v_mean [L,D,T], unc_R_diag_blocks [L,T,D*(D+1)/2], unc_R_offdiag_blocks [L,T-1,D,D]
        Output shapes: eta_1 [L,D,T], eta_2_diag_blocks [L,T,D,D], eta_2_offdiag_blocks [L,T-1,D,D]
        """

        # Constrain R
        R_diag_blocks, R_offdiag_blocks = jit(vmap(constrain_R, in_axes=(0,0)))(unc_R_diag_blocks,
                                                                                unc_R_offdiag_blocks)
        
        # Compute S
        S_diag_blocks, S_offdiag_blocks = jit(vmap(self.R_to_S, in_axes=(0,0)))(R_diag_blocks, # [L,M,D,D]
                                                                                R_offdiag_blocks) # [L,M-1,D,D]

        # Compute \bmeta_1
        eta_1 = jnp.einsum('ijkl,ilj->ikj', S_diag_blocks, v_mean) # [L,D,M]
        eta_1 = eta_1.at[:,:,1:].add(jnp.einsum('ijkl,ilj->ikj', S_offdiag_blocks, v_mean[:,:,:-1])) # [L,D,M]
        eta_1 = eta_1.at[:,:,:-1].add(jnp.einsum('ijlk,ilj->ikj', S_offdiag_blocks, v_mean[:,:,1:])) # [L,D,M]

        # Compute \bmeta_2, diagonal and off-diagonal blocks
        eta_2_diag_blocks = -0.5 * S_diag_blocks # [L,M,D,D]
        eta_2_offdiag_blocks = -0.5 * S_offdiag_blocks # [L,M-1,D,D]

        return eta_1, eta_2_diag_blocks, eta_2_offdiag_blocks

    def nat_to_params(
        self, 
        eta_1: f64['L D T'], 
        eta_2_diag_blocks: f64['L T D D'], 
        eta_2_offdiag_blocks: f64['L T-1 D D'] 
    ) -> tuple[f64['L D T'], f64['L T D*(D+1)/2'], f64['L T-1 D D']]:
        """
        Converts the natural parameters to their corresponding unconstrained variational parameters.

        Note: m = solve((RR^Tm = eta_1), m), R = S_to_R(S)

        Input shapes: eta_1 [L,D,T], eta_2_diag_blocks [L,T,D,D], eta_2_offdiag_blocks [L,T-1,D,D],
        Output shapes: v_mean [L,D,T], unc_R_diag_blocks [L,T,D*(D+1)/2], unc_R_offdiag_blocks [L,T-1,D,D]
        """

        S_diag_blocks = -2 * eta_2_diag_blocks # [L,T,D(D+1)/2]
        S_offdiag_blocks = -2 * eta_2_offdiag_blocks # [L,T-1,D,D]

        # Compute unconstrained log-Cholesky blocks
        unc_R_diag_blocks, unc_R_offdiag_blocks = jit(vmap(self.S_to_R, in_axes=(0,0)))(S_diag_blocks, 
                                                                                        S_offdiag_blocks)

        # Solve block tridiagonal system to find the variational mean
        v_mean = jit(vmap(self.solve_tridiag_system, in_axes=(0,0,0)))(eta_1, S_diag_blocks, S_offdiag_blocks) # [L,D,M]

        return v_mean, unc_R_diag_blocks, unc_R_offdiag_blocks
    
    def params_to_exp(
        self, 
        v_mean: f64['L D T'], 
        unc_R_diag_blocks: f64['L T D*(D+1)/2'], 
        unc_R_offdiag_blocks: f64['L T-1 D D']
    ) -> tuple[f64['L D T'], f64['L T D D'], f64['L T-1 D D']]:
        """
        Converts the unconstrained variational parameters to their corresponding expectation parameters.
        
        Note: xi_1 = m, xi_2 = BTD(mm^T + (RR^T)^{-1})

        Input shapes: v_mean [L,D,T], unc_R_diag_blocks [L,T,D*(D+1)/2], unc_R_offdiag_blocks [L,T-1,D,D]
        Output shapes: xi_1 [L,D,T], xi_2_diag_blocks [L,T,D,D], xi_2_offdiag_blocks [L,T-1,D,D]
        """

        # Constrain R
        R_diag_blocks, R_offdiag_blocks = jit(vmap(constrain_R, in_axes=(0,0)))(unc_R_diag_blocks,
                                                                                unc_R_offdiag_blocks)

        # Compute tridiagonal blocks of S
        S_diag_blocks, S_offdiag_blocks = jit(vmap(self.R_to_S, in_axes=(0,0)))(R_diag_blocks, # [L,T,D,D]
                                                                                R_offdiag_blocks) # [L,T-1,D,D]

        # Compute tridiagonal blocks of S^{-1}
        # Note: S_inv_diag_blocks [L,T,D,D], S_offdiag_blocks [L,T-1,D,D]
        S_inv_diag_blocks, S_inv_offdiag_blocks = jit(vmap(self.tridiag_inv, in_axes=(0,0)))(S_diag_blocks, 
                                                                                             S_offdiag_blocks)

        # Compute outer product of mean
        p_outer = jit(vmap(vmap(jnp.outer, in_axes=(1,1)), in_axes=(0,0)))
        outer_diag_blocks = p_outer(v_mean, v_mean) # [L,T,D,D]
        outer_offdiag_blocks = p_outer(v_mean[:,:,1:], v_mean[:,:,:-1]) # [L,T-1,D,D]

        xi_1 = v_mean # [L,D,T]
        xi_2_diag_blocks = outer_diag_blocks + S_inv_diag_blocks # [L,T,D,D]
        xi_2_offdiag_blocks = outer_offdiag_blocks + S_inv_offdiag_blocks # [L,T-1,D,D]

        return xi_1, xi_2_diag_blocks, xi_2_offdiag_blocks

    def exp_to_params(
        self, 
        xi_1: f64['L D T'], 
        xi_2_diag_blocks: f64['L T D D'], 
        xi_2_offdiag_blocks: f64['L T-1 D D']
    ) -> tuple[f64['L D T'], f64['L T D*(D+1)/2'], f64['L T-1 D D']]:
        """
        Converts the expectation parameters to their corresponding unconstrained variational parameters.

        Note: m = xi_1, BTD(R) = BTD(chol(S))

        For block-banded Cholesky decomposition from the inverse, we use the algorithm in (Asif and Moura, 2005). 

        Input shapes: xi_1 [L,D,T], xi_2_diag_blocks [L,T,D,D], xi_2_offdiag_blocks [L,T-1,D,D]
        Output shapes: v_mean [L,D,T], unc_R_diag_blocks [L,T,D*(D+1)/2], unc_R_offdiag_blocks [L,T-1,D,D] 
        """

        # Compute outer product of mean
        p_outer = jit(vmap(vmap(jnp.outer, in_axes=(1,1)), in_axes=(0,0)))
        outer_diag_blocks = p_outer(xi_1, xi_1) # [L,T,D,D]
        outer_offdiag_blocks = p_outer(xi_1[:,:,1:], xi_1[:,:,:-1]) # [L,T-1,D,D]

        S_inv_diag_blocks = xi_2_diag_blocks - outer_diag_blocks # [L,T,D,D]
        S_inv_offdiag_blocks = xi_2_offdiag_blocks - outer_offdiag_blocks # [L,T-1,D,D]

        @partial(vmap, in_axes=(0,0))
        def compute_diag_blocks(S_i_inv_diag_blocks, S_i_inv_offdiag_blocks):
            """Computes diagonal blocks of constrained R."""

            diag_final = S_i_inv_diag_blocks[-1,:,:] # [D,D]
            R_i_diag_final = jnp.linalg.cholesky(inv(diag_final, jitter=self.jitter)) # [D,D]

            def diag_helper(prev, idx):
                S_i_inv_diag_current = S_i_inv_diag_blocks[idx,:,:] # [D,D]
                S_i_inv_diag_next = S_i_inv_diag_blocks[idx+1,:,:] # [D,D]
                S_i_inv_offdiag = S_i_inv_offdiag_blocks[idx,:,:] # [D,D]

                schur = S_i_inv_diag_current - quadratic(S_i_inv_offdiag, S_i_inv_diag_next, jitter=self.jitter)
                schur_inverse = inv(schur, jitter=self.jitter)
                R_i_diag_block = jnp.linalg.cholesky(schur_inverse + self.jitter * jnp.eye(self.D))

                return prev, R_i_diag_block

            idxs = jnp.arange(0, self.T-1) # [0,...,T-2]
            dummy_init = jnp.zeros((self.D,self.D)) # [D,D]
            _, R_i_diag_blocks = lax.scan(diag_helper, dummy_init, idxs) # [M-1,D,D]
            R_i_diag_blocks = jnp.concatenate([R_i_diag_blocks, R_i_diag_final[None,:,:]], axis=0) # [M,D,D]

            return R_i_diag_blocks

        @partial(vmap, in_axes=(0,0,0))
        def compute_offdiag_blocks(S_i_inv_diag_blocks, S_i_inv_offdiag_blocks, R_i_diag_blocks):
            """Computes the off-diagonal blocks of constrained R."""

            def offdiag_helper(prev, idx):
                S_i_inv_diag = S_i_inv_diag_blocks[idx+1,:,:] # [D,D]
                S_i_inv_offdiag = S_i_inv_offdiag_blocks[idx,:,:] # [D,D]
                R_i_diag = R_i_diag_blocks[idx,:,:] # [D,D]

                R_i_offdiag_block = -inv_product(S_i_inv_offdiag, S_i_inv_diag, jitter=self.jitter) @ R_i_diag # [D,D]

                return prev, R_i_offdiag_block

            idxs = jnp.arange(0, self.T-1) # [0,...,T-2]
            dummy_init = jnp.zeros((self.D,self.D)) # [D,D]
            _, R_i_offdiag_blocks = lax.scan(offdiag_helper, dummy_init, idxs) # [M-1,D,D]

            return R_i_offdiag_blocks

        R_diag_blocks = compute_diag_blocks(S_inv_diag_blocks, S_inv_offdiag_blocks) # [L,M,D,D]
        R_offdiag_blocks = compute_offdiag_blocks(S_inv_diag_blocks, S_inv_offdiag_blocks, R_diag_blocks) # [L,M-1,D,D]

        # Unconstrain R
        unc_R_diag_blocks, unc_R_offdiag_blocks = jit(vmap(unconstrain_R, in_axes=(0,0)))(R_diag_blocks,
                                                                                          R_offdiag_blocks)

        # Mean parameter
        v_mean = xi_1 # [L,D,M]

        return v_mean, unc_R_diag_blocks, unc_R_offdiag_blocks

    def transition(
        self, 
        Z1: f64['L D'], 
        cov_infty: f64['L D D'], 
        times: Union[f64['T'], f64['T 1']], 
        key: chex.PRNGKey
    ) -> f64['L D T']:
        """Sample subsequent Z's given Z1."""

        T = times.shape[0]
        time_diffs = times[1:] - times[:-1]
        
        # Compute the state-transition matrix
        F = jit(vmap(self.compute_F))(self.model_params['unc_lengthscales'])
        
        def transition_helper(Z_prev, current):
            subkey, time_diff = current
            
            # Compute mean
            psi_k = jit(vmap(compute_psi, in_axes=(None,0)))(time_diff, F) # [L,D,D]
            Z_k_mean = jnp.einsum('ijk,ik->ij', psi_k, Z_prev) # [L,D,D], [L,D] -> [L,D]

            # Compute transition covariance
            phi_k = jit(vmap(cov_steady, in_axes=(None,0,0)))(time_diff, F, cov_infty) # [L,D,D]
            
            # Compute Cholesky decomposition for phi_k for more efficient sampling
            chol_k = jit(vmap(jnp.linalg.cholesky))(phi_k) # [L,D,D]
            
            # Sample from distribution
            epsilon = tfd.Normal(loc=0., scale=1.).sample((self.L,self.D), seed=subkey)
            Z_k = Z_k_mean + jnp.einsum('ijk,ik->ij', chol_k, epsilon) # [L,D]
            
            return Z_k, Z_k
        
        subkeys = jr.split(key, T-1)
        _, Z = lax.scan(transition_helper, Z1, (subkeys, time_diffs)) # [T-1,L,D]
        Z = jnp.vstack((Z1[None,:],Z)) # [T,L,D]
        Z = jnp.transpose(Z, axes=[1,2,0]) # [L,D,T]
            
        return Z
    
    def emission(
        self, 
        Z: f64['L D T'], 
        key: chex.PRNGKey
    ) -> f64['P T']:
        """Samples Y given W and Z."""

        Y_mean = self.model_params['W'] @ Z[:,0,:] # [P,L] * [L,T] -> [P,T]
        
        _, subkey = jr.split(key)
        std_normal = jnp.sqrt(softplus(self.model_params['unc_var'], low=EPS)) * jr.normal(subkey, Y_mean.shape)
        Y = Y_mean + std_normal
        
        return Y
    
    def sample(
        self, 
        times: Union[f64['T'], f64['T 1']], 
        key: chex.PRNGKey
    ) -> tuple[f64['L D T'], f64['P T']]:
        """Generates samples from FSDE."""

        # Sample initial state from steady-state covariance
        cov_infty = self.compute_cov_infty(self.model_params['unc_k_vars'], 
                                           self.model_params['unc_lengthscales']) # [L,D,D]
        chol_cov_infty = jit(vmap(jnp.linalg.cholesky))(cov_infty) # [L,D,D]

        _, subkey = jr.split(key)
        epsilon = tfd.Normal(loc=0., scale=1.).sample((self.L,self.D), seed=subkey) # [L,D]
        Z1 = jnp.einsum('ijk,ik->ij', chol_cov_infty, epsilon) # [L,D]
        
        # Transition
        Z = self.transition(Z1, cov_infty, times, key) # [L,D,T]
        
        # Emission
        Y = self.emission(Z, key) # [P,T]
        
        return Z, Y
    
    def compute_Lambda(
        self,
        psi_i: f64['T-1 D D'], 
        phi_i: f64['T-1 D D'], 
        cov_infty_i: f64['D D']
    ):
        """Computes the prior precision matrix blocks for the inducing points of each latent GP."""

        # Compute diagonal blocks
        T1 = jit(vmap(inv, in_axes=(0,None)))(phi_i, self.jitter) # [T-1,D,D]
        T2 = jit(vmap(quadratic, in_axes=(0,0)))(psi_i, phi_i) # [T-1,D,D]
        
        Lambda_diag_blocks = jnp.zeros((self.T,self.D,self.D)) # [M,D,D]
        Lambda_diag_blocks = Lambda_diag_blocks.at[0].add(inv(cov_infty_i, jitter=self.jitter))
        Lambda_diag_blocks = Lambda_diag_blocks.at[1:].add(T1)
        Lambda_diag_blocks = Lambda_diag_blocks.at[:-1].add(T2)
        
        # Compute lower off-diagonal blocks
        Lambda_offdiag_blocks = -jit(vmap(inv_product, in_axes=(0,0)))(psi_i, phi_i) # [T-1,D,D]
        
        return Lambda_diag_blocks, Lambda_offdiag_blocks
    
    def tridiag_inv(
        self, 
        diag_blocks: f64['T D D'], 
        offdiag_blocks: f64['T-1 D D'], 
        cholesky: Optional[bool] = False
    ):
        """
        Computes the diagonal and off-diagonal blocks of the inverse of a block-tridiagonal matrix.

        Implements the block-tridiagonal inversion algorithm in (Reuter and Hill, 2012).
        
        Shapes: diag_blocks [T,D,D], offdiag_blocks [T-1,D,D]
        """

        blocks = (diag_blocks, offdiag_blocks)
        
        # Compute Schur complements
        def schur_helper_A(A_prev, k, blocks=blocks):
            diag_blocks, offdiag_blocks = blocks
            
            chol_f = lambda: quadratic(offdiag_blocks[k], diag_blocks[k+1] - A_prev, jitter=self.jitter)
            inv_f = lambda: offdiag_blocks[k].T @ inv(diag_blocks[k+1] - A_prev, jitter=self.jitter) @ \
                            offdiag_blocks[k]

            A_k = lax.cond(cholesky, chol_f, inv_f)
            
            return A_k, A_k
        
        def schur_helper_B(B_prev, k, blocks=blocks):
            diag_blocks, offdiag_blocks = blocks

            chol_f = lambda: quadratic(offdiag_blocks[k-1].T, diag_blocks[k-1] - B_prev, jitter=self.jitter)
            inv_f = lambda: offdiag_blocks[k-1] @ inv(diag_blocks[k-1] - B_prev, jitter=self.jitter) @ \
                            offdiag_blocks[k-1].T
            
            B_k = lax.cond(cholesky, chol_f, inv_f)
            
            return B_k, B_k
        
        @partial(vmap, in_axes=(0,0,0,0))
        def offdiag_helper(diag_block, A, offdiag_block, inv_diag_block):
            result = -inv(diag_block - A, jitter=self.jitter) @ offdiag_block @ inv_diag_block
            
            return result
        
        A_T = jnp.zeros((self.D,self.D))
        _, A = lax.scan(schur_helper_A, A_T, jnp.arange(0,self.T-1)[::-1]) # A = [T-1,D,D], index starts at T-2
        A = jnp.flip(A, axis=0) # Order from A_1, ...., A_{T-1}
        A = jnp.concatenate((A, A_T[None,:]), axis=0) # [T,D,D]
        
        B_1 = jnp.zeros((self.D,self.D))
        _, B = lax.scan(schur_helper_B, B_1, jnp.arange(1,self.T)) # B = [T-1,D,D], index starts at 1 
        B = jnp.concatenate((B_1[None,:], B), axis=0) # [T,D,D]
        
        # Compute main diagonal blocks
        inv_diag_blocks = jit(vmap(inv, in_axes=(0,None)))(diag_blocks - A - B, self.jitter) # [T,D,D]
        
        # Compute lower off-diagonal blocks
        inv_offdiag_blocks = offdiag_helper(diag_blocks[1:], A[1:], offdiag_blocks, inv_diag_blocks[:-1]) # [T-1,D,D]

        return inv_diag_blocks, inv_offdiag_blocks
    
    def tridiag_logdet(
        self, 
        diag_blocks: f64['T D D'], 
        offdiag_blocks: f64['T-1 D D'], 
        cholesky: Optional[bool] = False
    ) -> f64:
        """
        Computes the log-determinant of a block-tridiagonal matrix.

        Implements the log-determinant algorithm in (Salkuyeh, 2006).

        Shapes: diag_blocks [T,D,D], offdiag_blocks [T-1,D,D]
        """

        blocks = (diag_blocks, offdiag_blocks)
        
        def det_helper(A_prev, k, blocks=blocks):
            diag_blocks, offdiag_blocks = blocks
            
            chol_f = lambda: diag_blocks[k] - quadratic(offdiag_blocks[k-1].T, A_prev, jitter=self.jitter)
            inv_f = lambda: diag_blocks[k] - offdiag_blocks[k-1] @ inv(A_prev, jitter=self.jitter) @ \
                            offdiag_blocks[k-1].T

            A_k = lax.cond(cholesky, chol_f, inv_f)
            logdet_k = jnp.multiply(*jnp.linalg.slogdet(A_k))
            
            return A_k, logdet_k
            
        _, logdets = lax.scan(det_helper, diag_blocks[0], jnp.arange(1,self.T))
        logdet = jnp.multiply(*jnp.linalg.slogdet(diag_blocks[0])) + jnp.sum(logdets)
            
        return logdet

    @partial(vmap, in_axes=(None,0,0,0,0,0,0,0))
    def KL(
        self,
        v_mean_i: f64['D T'], 
        Lambda_i_diag_blocks: f64['T D D'], 
        Lambda_i_offdiag_blocks:f64['T-1 D D'], 
        R_i_diag_blocks: f64['T D*(D+1)/2'], 
        R_i_offdiag_blocks: f64['T-1 D D'],
        S_i_inv_diag_blocks: f64['T D D'], 
        S_i_inv_offdiag_blocks: f64['T-1 D D']
    ) -> f64:
        """Computes KL(q||p) for variational distribution."""

        def m_Lambda_m(
            v_mean_i: f64['D T'], 
            Lambda_i_diag_blocks: f64['T D D'], 
            Lambda_i_offdiag_blocks: f64['T-1 D D']
        ) -> f64:
            
            @jit
            @partial(vmap, in_axes=(1,0,1))
            def helper(A, B, C):
                return jnp.squeeze(A.T @ B @ C)
            
            result = jnp.sum(helper(v_mean_i, Lambda_i_diag_blocks, v_mean_i))
            result += 2 * jnp.sum(helper(v_mean_i[:,1:], Lambda_i_offdiag_blocks, v_mean_i[:,:-1]))
            
            return result
        
        # Convert to triangular matrix
        S_i_diag_blocks, S_i_offdiag_blocks = self.R_to_S(R_i_diag_blocks, R_i_offdiag_blocks)
        
        kl = 0.5 * self.tridiag_logdet(S_i_diag_blocks, S_i_offdiag_blocks)
        kl += -0.5 * self.tridiag_logdet(Lambda_i_diag_blocks, Lambda_i_offdiag_blocks)
        kl += -0.5 * self.T * self.D
        kl += 0.5 * m_Lambda_m(v_mean_i, Lambda_i_diag_blocks, Lambda_i_offdiag_blocks)
        kl += 0.5 * (jnp.sum(Lambda_i_diag_blocks * S_i_inv_diag_blocks) + \
                     jnp.sum(Lambda_i_offdiag_blocks * S_i_inv_offdiag_blocks))
        
        return kl
    
    def compute_ELBO(
        self, 
        model_params: Dict,
        v_params: Dict
    ) -> f64:
        """Computes the ELBO for the training data."""
        
        # Compute steady-state covariance
        cov_infty = self.compute_cov_infty(model_params['unc_k_vars'], 
                                           model_params['unc_lengthscales']) # [L,D,D]

        # Constrain R
        R_diag_blocks, R_offdiag_blocks = jit(vmap(constrain_R, in_axes=(0,0)))(v_params['unc_R_diag_blocks'],
                                                                                v_params['unc_R_offdiag_blocks'])
        
        @partial(vmap, in_axes=(1,0))
        def trace_W_Cov(W_i: f64['P'], S_i_inv_diag_blocks: f64['T D D']) -> f64:
            return jnp.linalg.norm(W_i)**2 * jnp.sum(S_i_inv_diag_blocks[:,0,0])
        
        time_diffs = self.train_dataset.times[1:] - self.train_dataset.times[:-1]
        
        # Compute state-transition matrix
        F = jit(vmap(self.compute_F))(model_params['unc_lengthscales'])

        # Compute psi and phi
        prior_psi = jit(vmap(vmap(compute_psi, in_axes=(0,None)), in_axes=(None,0)))(time_diffs, F) # [L,T-1,D,D]
        prior_phi = jit(vmap(vmap(cov_steady, in_axes=(0,None,None)),
                             in_axes=(None,0,0)))(time_diffs, F, cov_infty) # [L,T-1,D,D]

        # Computes the precision and covariance blocks for all latent SDEs
        # Note: Lambda_diag_blocks: [L,T,D,D], Lambda_offdiag_blocks: [L,T-1,D,D]
        Lambda_diag_blocks, Lambda_offdiag_blocks = jit(vmap(self.compute_Lambda, in_axes=(0,0,0)))(prior_psi, 
                                                                                                    prior_phi, 
                                                                                                    cov_infty) 
        
        # Compute S from R
        # Note: S_diag_blocks [L,T,D,D], S_offdiag_blocks [L,T-1,D,D]
        S_diag_blocks, S_offdiag_blocks = jit(vmap(self.R_to_S, in_axes=(0,0)))(R_diag_blocks, 
                                                                                R_offdiag_blocks)
        
        # Compute inverse blocks of S
        # Note: S_inv_diag_blocks [L,T,D,D], S_offdiag_blocks [L,T-1,D,D]
        S_inv_diag_blocks, S_inv_offdiag_blocks = jit(vmap(self.tridiag_inv, in_axes=(0,0)))(S_diag_blocks, 
                                                                                             S_offdiag_blocks)
        
        # Get predictions
        Y_hat = model_params['W'] @ v_params['v_mean'][:,0,:] # [P,T]
        var = softplus(model_params['unc_var'], low=EPS)
        
        # Expected log-likelihood
        exp_loglik = -0.5 * self.T * self.P * jnp.log(2 * jnp.pi * var)
        exp_loglik += -0.5 * (1./var) * jnp.linalg.norm(self.train_dataset.Y - Y_hat)**2
        exp_loglik += -0.5 * (1./var) * jnp.sum(trace_W_Cov(model_params['W'], S_inv_diag_blocks))

        # KL terms
        kl = jnp.sum(self.KL(v_params['v_mean'], Lambda_diag_blocks, Lambda_offdiag_blocks,
                             R_diag_blocks, R_offdiag_blocks, S_inv_diag_blocks, S_inv_offdiag_blocks))

        elbo = exp_loglik - kl
        
        return (elbo, exp_loglik, kl)
    
    def loss(
        self, 
        model_params: Dict,
        v_params: Dict
    ) -> f64:
        """Compute the negative ELBO for the training data."""

        elbo, exp_loglik, kl = self.compute_ELBO(model_params, v_params)

        return -elbo, (exp_loglik, kl)
    
    def fit(
        self,
        n_steps: int,
        lr: float, 
        lr_steps: int, 
        log_rate: Optional[int] = 1,
        use_natgrad: Optional[bool] = True
    ) -> list[f64]:
        """Fits the FSDE with Adam and/or NGD."""

        # Initialize optimizers
        model_optimizer = optax.adam(lr['model_lr'])
        opt_state = model_optimizer.init(self.model_params)

        if use_natgrad:
            # Log-linear increase scheduler
            scheduler = loglinear_schedule(init_value=lr['var_lr_init'], 
                                           end_value=lr['var_lr_end'], 
                                           transition_steps=lr_steps)

            v_optimizer = optax.chain(optax.scale_by_schedule(scheduler),
                                      optax.scale(-1.0))
        else:
            v_optimizer = optax.adam(lr['var_adam_lr'])

        v_opt_state = v_optimizer.init(self.v_params)

        # Collect initial setting
        init_state = (self.model_params, self.v_params, opt_state, v_opt_state)

        # Track total training time
        start_time = datetime.now()

        # Single optimization step
        @progress_bar_scan2(n_steps, log_rate)
        def step(carry, current):
            # Unpack
            model_params, v_params, opt_state, v_opt_state = carry
            step_idx = current

            # Compute gradients
            out, grads = jit(value_and_grad(self.loss, argnums=(0,1), has_aux=True))(model_params, 
                                                                                     v_params)
            elbo = -out[0]
            exp_loglik = out[1][0]
            kl = out[1][1]

            # Update model parameters via ADAM
            p_clip_grads = Partial(clip_grads, max_norm=1e4)
            clipped_grads = tree_map(p_clip_grads, grads[0])

            model_updates, opt_state = model_optimizer.update(clipped_grads, opt_state)
            model_params = optax.apply_updates(model_params, model_updates)

            # Update variational parameters via NGD
            def natgrad_update(v_params, v_opt_state):
                v_grads = (grads[1]['v_mean'].astype(jnp.float64), 
                           grads[1]['unc_R_diag_blocks'].astype(jnp.float64),
                           grads[1]['unc_R_offdiag_blocks'].astype(jnp.float64))

                # Compute gradient wrt expectation parameters
                exp_params = self.params_to_exp(v_params['v_mean'].astype(jnp.float64),
                                                v_params['unc_R_diag_blocks'].astype(jnp.float64),
                                                v_params['unc_R_offdiag_blocks'].astype(jnp.float64))

                _, exp_vjp = vjp(self.exp_to_params, *exp_params)
                grads_exp = exp_vjp(v_grads)

                # Compute natural gradient wrt parameters
                nat_params = self.params_to_nat(v_params['v_mean'].astype(jnp.float64),
                                                v_params['unc_R_diag_blocks'].astype(jnp.float64),
                                                v_params['unc_R_offdiag_blocks'].astype(jnp.float64))

                _, natgrads = jvp(self.nat_to_params, nat_params, grads_exp)

                p_clip_grads = Partial(clip_grads, max_norm=1e4)
                clipped_natgrads = tree_map(p_clip_grads, natgrads)

                natgrad_dict = dict(v_mean=clipped_natgrads[0],
                                    unc_R_diag_blocks=clipped_natgrads[1],
                                    unc_R_offdiag_blocks=clipped_natgrads[2])

                v_updates, v_opt_state = v_optimizer.update(natgrad_dict, v_opt_state)
                v_params = optax.apply_updates(v_params, v_updates)

                return v_params, v_opt_state

            # Update variational parameters via Adam
            def adam_update(v_params, v_opt_state):
                p_clip_grads = Partial(clip_grads, max_norm=1e4)
                clipped_v_grads = tree_map(p_clip_grads, grads[1])

                v_updates, v_opt_state = v_optimizer.update(clipped_v_grads, v_opt_state)
                v_params = optax.apply_updates(v_params, v_updates)

                return v_params, v_opt_state

            v_params, v_opt_state = lax.cond(use_natgrad, 
                                             natgrad_update, 
                                             adam_update, 
                                             v_params,
                                             v_opt_state)

            # Collect updates
            carry = (model_params, v_params, opt_state, v_opt_state)

            return carry, (elbo, exp_loglik, kl)

        # Optimize the model
        step_idxs = jnp.arange(0, n_steps)
        (model_params, v_params, _, _), metrics = lax.scan(step, init_state, step_idxs)

        train_time = datetime.now() - start_time
        print(f'[Total training time] {str(train_time).split(".")[0]}')

        # Unpack parameters
        self.model_params = model_params
        self.v_params = v_params

        return metrics

    def past_moments(
        self, 
        args: Dict
    ) -> tuple[f64['D'], f64['D D']]:
        """Computes the approximate moments for past time points."""

        # Unpack arguments
        past_time = args['time']
        cov_infty_i = args['cov_infty_i']
        F_i = args['F_i']
        v_mean_i = args['v_mean_i']
        S_i_inv_diag_blocks = args['S_i_inv_diag_blocks']

        # Compute the time difference
        time_diff = self.first_train_time - past_time

        psi = compute_psi(time_diff, F_i)
        phi = cov_steady(time_diff, F_i, cov_infty_i)

        chol_infty_i = jnp.linalg.cholesky(cov_infty_i + self.jitter * jnp.eye(self.D)) # [D,D]
        chol_factor_i = jscipy.linalg.solve_triangular(chol_infty_i, psi @ cov_infty_i, lower=True) # [D,D]
        factor_i = jscipy.linalg.solve_triangular(chol_infty_i.T, chol_factor_i, lower=False) # [D,D]

        # Variational parameters of the first inducing point
        v_mean_i_first = lax.dynamic_slice(v_mean_i, (0,0), (self.D,1)) # [D,1]
        S_i_inv_diag = lax.dynamic_slice(S_i_inv_diag_blocks, (0,0,0), (1,self.D,self.D)).squeeze() # [D,D]

        # Compute approximate mean and covariance
        approx_mean = (factor_i.T @ v_mean_i_first).squeeze() # [D,1] -> [D,]
        approx_cov = factor_i.T @ S_i_inv_diag @ factor_i + cov_infty_i - chol_factor_i.T @ chol_factor_i # [D,D]

        return approx_mean, approx_cov

    def smooth_moments(
        self, 
        args: Dict
    ) -> tuple[f64['D'], f64['D D']]:
        """Computes the approximate moments for in-between time points."""

        # Unpack arguments
        smooth_time = args['time']
        cov_infty_i = args['cov_infty_i']
        F_i = args['F_i']
        v_mean_i = args['v_mean_i']
        S_i_inv_diag_blocks = args['S_i_inv_diag_blocks']
        S_i_inv_offdiag_blocks = args['S_i_inv_offdiag_blocks']

        train_times = self.train_dataset.times.squeeze()

        # Find closest two inducing points
        lower_idx = (jnp.where(train_times < smooth_time, 1, 0).argmin() - 1).astype(jnp.int64)
        upper_idx = (jnp.where(train_times > smooth_time, 1, 0).argmax()).astype(jnp.int64)

        lower = lax.dynamic_slice(train_times, (lower_idx,), (1,)).squeeze()
        upper = lax.dynamic_slice(train_times, (upper_idx,), (1,)).squeeze()

        v_mean_i_lower = lax.dynamic_slice(v_mean_i, (0,lower_idx), (self.D,1)) # [D,1]
        v_mean_i_upper = lax.dynamic_slice(v_mean_i, (0,upper_idx), (self.D,1)) # [D,1]
        v_mean_i_pair = jnp.vstack((v_mean_i_lower, v_mean_i_upper)) # [2*D,1]

        S_i_inv_diag_pair = lax.dynamic_slice(S_i_inv_diag_blocks, (lower_idx,0,0), (2,self.D,self.D)).squeeze() # [2,D,D]
        S_i_inv_offdiag = lax.dynamic_slice(S_i_inv_offdiag_blocks, (lower_idx,0,0), (1,self.D,self.D)).squeeze() # [D,D]
        S_i_inv_pair = jscipy.linalg.block_diag(*S_i_inv_diag_pair) # [2*D,2*D]
        S_i_inv_pair = S_i_inv_pair.at[self.D:,:self.D].add(S_i_inv_offdiag) # [2*D,2*D]
        S_i_inv_pair = S_i_inv_pair.at[:self.D,self.D:].add(S_i_inv_offdiag.T) # [2*D,2*D]

        # Compute time differences
        upper_time_diff = upper - smooth_time # t_{k+1} - t_*
        lower_time_diff = smooth_time - lower # t_* - t_k

        # Compute required terms
        Sigma_lower = compute_psi(lower_time_diff, F_i) @ cov_infty_i # \bmSig_{*,k}, [D,D]
        Sigma_upper = cov_infty_i @ compute_psi(upper_time_diff, F_i).T # \bmSig_{*,k+1}, [D,D]
        
        phi = cov_steady(upper - lower, F_i, cov_infty_i) # \bmPhi_{k+1}, [D,D]
        psi = compute_psi(upper - lower, F_i) # \bmPsi_{k+1}, [D,D]
        
        phi_inv_psi_i = inv_product(psi, phi, jitter=self.jitter) # \bmPhi_{k+1}^{-1} \bmPsi_{k+1}, [D,D]
        psi_phi_inv_psi_i = quadratic(psi, phi, jitter=self.jitter) # \bmPsi_{k+1}^T \bmPhi_{k+1}^{-1} \bmPsi_{k+1}, [D,D]

        # (\bmSig_{*,k+1} - \bmSig_{*,k} @ \bmPsi_{k+1}.T) @ \bmPhi_{k+1}^{-1}
        upper_to_time = (Sigma_upper - Sigma_lower @ psi.T) @ inv(phi, jitter=self.jitter) # [D,D]
        lower_to_time = compute_psi(lower_time_diff, F_i) - upper_to_time @ psi # [D,D]
        mean_transform = jnp.hstack((lower_to_time, upper_to_time)) # [D,2*D]

        # Compute approximate mean
        approx_mean = (mean_transform @ v_mean_i_pair).squeeze() # [D,1] -> [D,]

        # Compute prior conditional covariance
        cov_prior = cov_steady(lower_time_diff, F_i, cov_infty_i) - \
                    quadratic(Sigma_upper.T, phi, jitter=self.jitter) + \
                    Sigma_upper @ phi_inv_psi_i @ Sigma_lower.T - \
                    Sigma_lower @ psi_phi_inv_psi_i @ Sigma_lower.T + \
                    Sigma_lower @ phi_inv_psi_i.T @ Sigma_upper.T

        # Compute approximate covariance
        approx_cov = cov_prior + mean_transform @ S_i_inv_pair @ mean_transform.T # [D,D]

        return approx_mean, approx_cov

    def forecast_moments(
        self, 
        args: Dict
    ) -> tuple[f64['D'], f64['D D']]:
        """Computes the approximate moments for future time points."""

        # Unpack arguments
        forecast_time = args['time']
        cov_infty_i = args['cov_infty_i']
        F_i = args['F_i']
        v_mean_i = args['v_mean_i']
        S_i_inv_diag_blocks = args['S_i_inv_diag_blocks']

        # Compute the time difference
        time_diff = forecast_time - self.last_train_time

        psi = compute_psi(time_diff, F_i)
        phi = cov_steady(time_diff, F_i, cov_infty_i)

        # Variational parameters for last inducing point
        v_mean_i_final = lax.dynamic_slice(v_mean_i, (0,-1), (self.D,1)) # [D,1]
        S_i_inv_diag = lax.dynamic_slice(S_i_inv_diag_blocks, (-1,0,0), (1,self.D,self.D)).squeeze() # [D,D]

        # Compute approximate mean and covariance
        approx_mean = (psi @ v_mean_i_final).squeeze() # [D,1] -> [D,]
        approx_cov = psi @ S_i_inv_diag @ psi.T + phi # [D,D]

        return approx_mean, approx_cov

    @partial(vmap, in_axes=(None,None,None,0,0,0,0,0))
    @partial(vmap, in_axes=(None,0,0,None,None,None,None,None))
    def approx_moments(
        self, 
        time: f64, 
        mode: i64, 
        cov_infty_i: f64['D D'], 
        F_i: f64['D D'], 
        v_mean_i: f64['D M'], 
        S_i_inv_diag_blocks: f64['M D D'], 
        S_i_inv_offdiag_blocks: f64['M-1 D D']
    ) -> tuple[f64['D'], f64['D D']]:
        """Computes the approximate moments for a given time point."""

        # Gather arguments
        args = dict(time=time,
                    cov_infty_i=cov_infty_i,
                    F_i=F_i,
                    v_mean_i=v_mean_i,
                    S_i_inv_diag_blocks=S_i_inv_diag_blocks,
                    S_i_inv_offdiag_blocks=S_i_inv_offdiag_blocks)

        # Call appropriate approximation call
        return lax.switch(mode, [self.past_moments, self.smooth_moments, self.forecast_moments], args)

    def predict(
        self, 
        times: Union[f64['V'], f64['V 1']], 
        F: f64['L D D'],  
        unc_var: f64, 
        W: f64['P L'], 
        cov_infty: f64['L D D'], 
        v_mean: f64['L D M'], 
        S_inv_diag_blocks: f64['L M D D'], 
        S_inv_offdiag_blocks: f64['L M-1 D D'], 
        full_result: Optional[bool] = False, 
        output_latent: Optional[bool] = False
    ):
        """
        Generate predictions for the given time points.
        
        Note: The output_latent flag makes the function return the approximate moments of the latent states.
        """

        times = times.squeeze()

        # Separate interpolation and extrapolation
        modes_1 = jnp.where(times < self.first_train_time, 0, 1)
        modes_2 = jnp.where(times > self.last_train_time, 1, 0)
        modes = modes_1 + modes_2

        # Compute approximate mean and covariance
        args = [times, modes, cov_infty, F, v_mean, S_inv_diag_blocks, S_inv_offdiag_blocks]
        approx_mean, approx_cov = jit(self.approx_moments)(*args)

        if output_latent:
            return approx_mean, approx_cov

        # Compute predicted mean
        Y_hat = W @ approx_mean[:,:,0] # [P,L] x [L,V] -> [P,V]

        def compute_pred_cov(approx_cov_t):
            """
            Computes the predictive variance.
            
            Note: approx_cov_t [L,D,D]
            """

            var = softplus(unc_var, low=EPS)
            scale = approx_cov_t[:,0,0] # [L,]
            scale_mat = jnp.diag(scale)
            pred_cov = W @ scale_mat @ W.T + var * jnp.eye(self.P) # [P,P]

            return pred_cov

        Y_cov = jit(vmap(compute_pred_cov, in_axes=1))(approx_cov) # [V,P,P]

        if full_result:
            prediction = {'approx_mean': approx_mean, 'approx_cov': approx_cov,
                          'Y_hat': Y_hat, 'Y_cov': Y_cov}

            return prediction

        else:
            return Y_hat, Y_cov

    def predict_train(self):
        """Computes the approximate mean and covariance for the training data."""

        Y_hat = self.model_params['W'] @ self.v_params['v_mean'][:,0,:]

        def compute_pred_cov(approx_cov_t):
            """
            Computes the predictive variance.
            
            Note: approx_cov_t [L,D,D]
            """

            var = softplus(self.model_params['unc_var'], low=EPS)
            W = self.model_params['W']
            scale = approx_cov_t[:,0,0] # [L,]
            scale_mat = jnp.diag(scale)
            pred_cov = W @ scale_mat @ W.T + var * jnp.eye(self.P) # [P,P]

            return pred_cov

        # Constrain R
        R_diag_blocks, R_offdiag_blocks = jit(vmap(constrain_R, in_axes=(0,0)))(self.v_params['unc_R_diag_blocks'],
                                                                                self.v_params['unc_R_offdiag_blocks'])

        # Compute S from R
        # Note: S_diag_blocks [L,T,D,D], S_offdiag_blocks [L,T-1,D,D]
        S_diag_blocks, S_offdiag_blocks = jit(vmap(self.R_to_S, in_axes=(0,0)))(R_diag_blocks, 
                                                                                R_offdiag_blocks)
        
        # Compute inverse blocks of S
        # Note: S_inv_diag_blocks [L,T,D,D], S_offdiag_blocks [L,T-1,D,D]
        S_inv_diag_blocks, S_inv_offdiag_blocks = jit(vmap(self.tridiag_inv, in_axes=(0,0)))(S_diag_blocks, 
                                                                                             S_offdiag_blocks)

        Y_cov = jit(vmap(compute_pred_cov, in_axes=1))(S_inv_diag_blocks) # [T,P,P]

        return Y_hat, Y_cov

@chex.dataclass
class FSDE_SVI:
    """Factorial SDE with Sparse Variational Inference (FSDE-SVI)"""

    model_params: Dict
    v_params: Dict
    kernel: Dict
    ind_times: chex.ArrayDevice
    num_times: i64
    compute_cov_infty: Callable
    compute_F: Callable
    jitter: f64

    @property
    def L(self) -> int:
        return self.model_params['W'].shape[1]

    @property
    def P(self) -> int:
        return self.model_params['W'].shape[0]

    @property
    def M(self) -> int:
        return self.ind_times.size

    @property
    def T(self) -> int:
        return self.num_times

    @property
    def D(self) -> int:
        return self.v_params['v_mean'].shape[1]

    @property
    def first_ind_time(self) -> f64:
        return self.ind_times[0]

    @property
    def last_ind_time(self) -> f64:
        return self.ind_times[-1]

    @property
    def params(self) -> Dict:
        return (self.model_params, self.v_params)

    def get_MOGP_params(self) -> Dict:
        """Returns the MOGP parameter values."""

        return self.model_params

    def get_v_params(self) -> Dict:
        """Returns the variational parameter values."""

        return self.v_params

    def precompute_pred_args(self):
        """Precomputes the arguments required for prediction."""

        # Compute state-transition matrix
        F = jit(vmap(self.compute_F))(self.model_params['unc_lengthscales'])

        # Compute steady-state covariance
        cov_infty = self.compute_cov_infty(self.model_params['unc_k_vars'], 
                                           self.model_params['unc_lengthscales']) # [L,D,D]

        # Constrain R
        R_diag_blocks, R_offdiag_blocks = jit(vmap(constrain_R, in_axes=(0,0)))(self.v_params['unc_R_diag_blocks'],
                                                                                self.v_params['unc_R_offdiag_blocks'])

        # Compute S from R
        # Note: S_diag_blocks [L,M,D,D], S_offdiag_blocks [L,M-1,D,D]
        S_diag_blocks, S_offdiag_blocks = jit(vmap(self.R_to_S, in_axes=(0,0)))(R_diag_blocks, R_offdiag_blocks)
        
        # Compute inverse blocks of S
        # Note: S_inv_diag_blocks [L,M,D,D], S_offdiag_blocks [L,M-1,D,D]
        S_inv_diag_blocks, S_inv_offdiag_blocks = jit(vmap(self.tridiag_inv, in_axes=(0,0)))(S_diag_blocks, 
                                                                                             S_offdiag_blocks)

        return (F, self.model_params['unc_var'], self.model_params['W'], cov_infty, 
                self.v_params['v_mean'], S_inv_diag_blocks, S_inv_offdiag_blocks)

    def R_to_S(
        self, 
        R_i_diag_blocks: f64['M D*(D+1)/2'], 
        R_i_offdiag_blocks: f64['M-1 D D']
    ) -> tuple[f64['M D D'], f64['M-1 D D']]:
        """
        Computes the variational precision matrix S from its constrained Cholesky factor R.
        
        Input shapes: R_i_diag_blocks [M,D*(D+1)/2], R_i_offdiag_blocks [M-1,D,D]
        Output shapes: S_i_diag_blocks [M,D,D], S_i_offdiag_blocks [M-1,D,D]       
        """

        # Convert diagonal blocks to lower triangular blocks
        triangular = tfb.FillTriangular(upper=False)
        R_i_diag_blocks = triangular.forward(R_i_diag_blocks) # [M,D,D]

        # Compute the diagonal blocks
        S_i_diag_blocks_1 = jnp.einsum('ijk,ilk->ijl', R_i_diag_blocks, R_i_diag_blocks) # [M,D,D]
        S_i_diag_blocks_2 = jnp.einsum('ijk,ilk->ijl', R_i_offdiag_blocks, R_i_offdiag_blocks) # [M-1,D,D]
        S_i_diag_blocks = S_i_diag_blocks_1 # [M,D,D]
        S_i_diag_blocks = S_i_diag_blocks.at[1:,:,:].add(S_i_diag_blocks_2) # [M,D,D]

        # Compute the off-diagonal blocks
        S_i_offdiag_blocks = jnp.einsum('ijk,ilk->ijl', R_i_offdiag_blocks, R_i_diag_blocks[:-1,:,:]) # [M-1,D,D]

        return S_i_diag_blocks, S_i_offdiag_blocks

    def S_to_R(
        self, 
        S_i_diag_blocks: f64['M D D'], 
        S_i_offdiag_blocks: f64['M-1 D D']
    ) -> tuple[f64['M D*(D+1)/2'], f64['M-1 D D']]:
        """
        Computes the unconstrained Cholesky factor R from the variational precision matrix S.
        
        Input shapes: S_i_diag_blocks [M,D,D], S_i_offdiag_blocks [M-1,D,D]
        Output shapes: unc_R_i_diag_blocks [M,D*(D+1)/2], unc_R_i_offdiag_blocks [M-1,D,D]
        """

        # Initial diagonal Cholesky block
        R_i_diag_block_1 = jnp.linalg.cholesky(S_i_diag_blocks[0,:,:]) # [D,D]

        # Computes \bmR_{\ell,i} and \bmR_{\ell,i,i-1} given \bmR_{\ell,i-1}
        def helper(R_i_diag_prev, idx):
            S_i_diag_block = lax.dynamic_slice(S_i_diag_blocks, (idx+1,0,0), (1,self.D,self.D)).squeeze()
            S_i_offdiag_block = lax.dynamic_slice(S_i_offdiag_blocks, (idx,0,0), (1,self.D,self.D)).squeeze()
            
            # Find \bmR_{\ell,i,i-1}
            R_i_offdiag_block = lax.linalg.triangular_solve(R_i_diag_prev.T, S_i_offdiag_block, left_side=False, lower=False)

            # Find \bmR_{\ell,i}
            res = S_i_diag_block - R_i_offdiag_block @ R_i_offdiag_block.T
            R_i_diag_block = jnp.linalg.cholesky(res)

            # Stack diagonal and off-diagonal blocks
            R_i_blocks = jnp.concatenate([R_i_diag_block[None,:,:], R_i_offdiag_block[None,:,:]]) # [2,D,D]

            return R_i_diag_block, R_i_blocks

        idxs = jnp.arange(0, self.M-1)
        _, R_i_blocks = lax.scan(helper, R_i_diag_block_1, idxs) # [M-1,2,D,D]

        # Diagonal blocks
        R_i_diag_blocks = R_i_blocks[:,0,:,:] # [M-1,D,D]
        R_i_diag_blocks = jnp.concatenate([R_i_diag_block_1[None,:,:],
                                           R_i_diag_blocks]) # [M,D,D]
        
        # Off-diagonal blocks
        R_i_offdiag_blocks = R_i_blocks[:,1,:,:] # [M-1,D,D]
        
        # Unconstrain R
        unc_R_i_diag_blocks, unc_R_i_offdiag_blocks = unconstrain_R(R_i_diag_blocks, R_i_offdiag_blocks)

        return unc_R_i_diag_blocks, unc_R_i_offdiag_blocks

    def solve_tridiag_system(
        self, 
        eta_i_1: f64['D M'], 
        S_i_diag_blocks: f64['M D D'], 
        S_i_offdiag_blocks: f64['M-1 D D']
    ) -> f64['D M']:
        """
        Solve the block-tridiagonal system of equations to recover the variational mean.

        Used in the nat_to_params() function.

        Input shapes: eta_i_1 [D,M], S_i_diag_blocks [M,D,D], S_i_offdiag_blocks [M-1,D,D]
        Output shapes: v_mean_i [D,M]
        """

        # STEP 1: Upper block-bidiagonalization
        eta_i_1_init = eta_i_1[:,0:1] # \bmeta_{\ell,1}^{(1)}, [D,1]
        S_i_diag_init = S_i_diag_blocks[0,:,:] # \bmS_{\ell,1}, [D,D]
        S_i_offdiag_init = S_i_offdiag_blocks[0,:,:] # \bmS_{\ell,21}, [D,D]
        
        off_init = inv_product(S_i_offdiag_init.T, S_i_diag_init, jitter=self.jitter) # [D,D]
        eta_init = inv_product(eta_i_1_init, S_i_diag_init, jitter=self.jitter) # [D,1]
        init = jnp.concatenate([off_init, eta_init], axis=-1) # [D,D+1]

        def bidiag_helper(prev, idx):
            # Unstack carry
            off_prev = prev[:,:-1] # [D,D]
            eta_prev = prev[:,-1:] # [D,1]

            eta_i_1_idx = lax.dynamic_slice(eta_i_1, (0,idx), (self.D,1)) # [D,1]
            S_i_diag_idx = lax.dynamic_slice(S_i_diag_blocks, (idx,0,0), (1,self.D,self.D)).squeeze() # [D,D]
            S_i_offdiag_idx = lax.dynamic_slice(S_i_offdiag_blocks, (idx-1,0,0), (1,self.D,self.D)).squeeze() # [D,D]
            S_i_offdiag_next = lax.dynamic_slice(S_i_offdiag_blocks, (idx,0,0), (1,self.D,self.D)).squeeze() # [D,D]

            # Compute \bmS_{\ell,i,i+1}^*
            res = S_i_diag_idx - S_i_offdiag_idx @ off_prev # \bmS_{\ell,i}^*, [D,D]
            off_new = inv_product(S_i_offdiag_next.T, res, jitter=self.jitter) # [D,D]

            # Compute \bmeta_{\ell,i}^{(1)**}
            res2 = eta_i_1_idx - S_i_offdiag_idx @ eta_prev # [D,1]
            eta_new = inv_product(res2, res, jitter=self.jitter) # [D,1]

            # Stack results
            stacked = jnp.concatenate([off_new, eta_new], axis=-1) # [D,D+1]

            return stacked, stacked

        idxs = jnp.arange(1, self.M-1)
        _, stacked = lax.scan(bidiag_helper, init, idxs) # [M-2,D,D+1]

        # Unstack result
        off_blocks = stacked[:,:,:-1] # [M-2,D,D]
        eta_blocks = stacked[:,:,-1:] # [M-2,D,1]

        # Compute last block
        eta_i_1_final = eta_i_1[:,-1:] # \bmeta_{\ell,M}^{(1)} # [D,1]
        S_i_diag_final = S_i_diag_blocks[-1,:,:] # \bmS_{\ell,M} # [D,D]
        S_i_offdiag_final = S_i_offdiag_blocks[-1,:,:] # \bmS_{\ell,M,M-1} # [D,D]
        off_final = S_i_diag_final - S_i_offdiag_final @ off_blocks[-1,:,:] # [D,D]
        eta_final = eta_i_1_final - S_i_offdiag_final @ eta_blocks[-1,:,:] # [D,1]

        off_blocks = jnp.concatenate([off_init[None,:,:], off_blocks, off_final[None,:,:]], axis=0) # [M,D,D]
        eta_blocks = jnp.concatenate([eta_init[None,:,:], eta_blocks, eta_final[None,:,:]], axis=0) # [M,D,1]

        # STEP 2: Back-substitution to find mean
        def back_sub_helper(v_mean_i_prev, idx):
            eta_block_idx = lax.dynamic_slice(eta_blocks, (idx,0,0), (1,self.D,1)).squeeze(axis=0) # [D,1]
            off_block_idx = lax.dynamic_slice(off_blocks, (idx,0,0), (1,self.D,self.D)).squeeze() # [D,D]

            v_mean_i_idx = (eta_block_idx - off_block_idx @ v_mean_i_prev) # [D,1]

            return v_mean_i_idx, v_mean_i_idx

        v_mean_i_final = inv_product(eta_blocks[-1,:,:], off_blocks[-1,:,:], jitter=self.jitter) # [D,1]
        idxs = jnp.flip(jnp.arange(0, self.M-1))
        _, v_mean_i = lax.scan(back_sub_helper, v_mean_i_final, idxs) # [M-1,D,1]
        v_mean_i = v_mean_i.squeeze(axis=-1) # [M-1,D]
        v_mean_i = jnp.flip(v_mean_i, axis=0).T # [D,M-1]
        v_mean_i = jnp.concatenate([v_mean_i, v_mean_i_final], axis=1) # [D,M]

        return v_mean_i

    def params_to_nat(
        self, 
        v_mean: f64['L D M'], 
        unc_R_diag_blocks: f64['L M D*(D+1)/2'], 
        unc_R_offdiag_blocks: f64['L M-1 D D']
    ) -> tuple[f64['L D M'], f64['L M D D'], f64['L M-1 D D']]:
        """
        Converts the unconstrained variational parameters to their corresponding natural parameters.

        Note: eta_1 = (RR^T)m, eta_2 = BTD(-0.5 * RR^T)

        Input shapes: v_mean [L,D,M], unc_R_diag_blocks [L,M,D*(D+1)/2], unc_R_offdiag_blocks [L,M-1,D,D]
        Output shapes: eta_1 [L,D,M], eta_2_diag_blocks [L,M,D,D], eta_2_offdiag_blocks [L,M-1,D,D]
        """

        # Constrain R
        R_diag_blocks, R_offdiag_blocks = jit(vmap(constrain_R, in_axes=(0,0)))(unc_R_diag_blocks,
                                                                                unc_R_offdiag_blocks)
        
        # Compute S
        S_diag_blocks, S_offdiag_blocks = jit(vmap(self.R_to_S, in_axes=(0,0)))(R_diag_blocks, # [L,M,D,D]
                                                                                R_offdiag_blocks) # [L,M-1,D,D]

        # Compute \bmeta_1
        eta_1 = jnp.einsum('ijkl,ilj->ikj', S_diag_blocks, v_mean) # [L,D,M]
        eta_1 = eta_1.at[:,:,1:].add(jnp.einsum('ijkl,ilj->ikj', S_offdiag_blocks, v_mean[:,:,:-1])) # [L,D,M]
        eta_1 = eta_1.at[:,:,:-1].add(jnp.einsum('ijlk,ilj->ikj', S_offdiag_blocks, v_mean[:,:,1:])) # [L,D,M]

        # Compute \bmeta_2, diagonal and off-diagonal blocks
        eta_2_diag_blocks = -0.5 * S_diag_blocks # [L,M,D,D]
        eta_2_offdiag_blocks = -0.5 * S_offdiag_blocks # [L,M-1,D,D]

        return eta_1, eta_2_diag_blocks, eta_2_offdiag_blocks

    def nat_to_params(
        self, 
        eta_1: f64['L D M'], 
        eta_2_diag_blocks: f64['L M D D'], 
        eta_2_offdiag_blocks: f64['L M-1 D D'] 
    ) -> tuple[f64['L D M'], f64['L M D*(D+1)/2'], f64['L M-1 D D']]:
        """
        Converts the natural parameters to their corresponding unconstrained variational parameters.

        Note: m = solve((RR^Tm = eta_1), m), R = S_to_R(S)

        Input shapes: eta_1 [L,D,M], eta_2_diag_blocks [L,M,D,D], eta_2_offdiag_blocks [L,M-1,D,D],
        Output shapes: v_mean [L,D,M], unc_R_diag_blocks [L,M,D*(D+1)/2], unc_R_offdiag_blocks [L,M-1,D,D]
        """

        S_diag_blocks = -2 * eta_2_diag_blocks # [L,M,D(D+1)/2]
        S_offdiag_blocks = -2 * eta_2_offdiag_blocks # [L,M-1,D,D]

        # Compute unconstrained log-Cholesky blocks
        unc_R_diag_blocks, unc_R_offdiag_blocks = jit(vmap(self.S_to_R, in_axes=(0,0)))(S_diag_blocks, 
                                                                                        S_offdiag_blocks)

        # Solve block tridiagonal system to find the variational mean
        v_mean = jit(vmap(self.solve_tridiag_system, in_axes=(0,0,0)))(eta_1, S_diag_blocks, S_offdiag_blocks) # [L,D,M]

        return v_mean, unc_R_diag_blocks, unc_R_offdiag_blocks

    def params_to_exp(
        self, 
        v_mean: f64['L D M'], 
        unc_R_diag_blocks: f64['L M D*(D+1)/2'], 
        unc_R_offdiag_blocks: f64['L M-1 D D']
    ) -> tuple[f64['L D M'], f64['L M D D'], f64['L M-1 D D']]:
        """
        Converts the unconstrained variational parameters to their corresponding expectation parameters.
        
        Note: xi_1 = m, xi_2 = BTD(mm^T + (RR^T)^{-1})

        Input shapes: v_mean [L,D,M], unc_R_diag_blocks [L,M,D*(D+1)/2], unc_R_offdiag_blocks [L,M-1,D,D]
        Output shapes: xi_1 [L,D,M], xi_2_diag_blocks [L,M,D,D], xi_2_offdiag_blocks [L,M-1,D,D]
        """

        # Constrain R
        R_diag_blocks, R_offdiag_blocks = jit(vmap(constrain_R, in_axes=(0,0)))(unc_R_diag_blocks,
                                                                                unc_R_offdiag_blocks)

        # Compute tridiagonal blocks of S
        S_diag_blocks, S_offdiag_blocks = jit(vmap(self.R_to_S, in_axes=(0,0)))(R_diag_blocks, # [L,M,D,D]
                                                                                R_offdiag_blocks) # [L,M-1,D,D]

        # Compute tridiagonal blocks of S^{-1}
        # Note: S_inv_diag_blocks [L,M,D,D], S_offdiag_blocks [L,M-1,D,D]
        S_inv_diag_blocks, S_inv_offdiag_blocks = jit(vmap(self.tridiag_inv, in_axes=(0,0)))(S_diag_blocks, 
                                                                                             S_offdiag_blocks)

        # Compute outer product of mean
        p_outer = jit(vmap(vmap(jnp.outer, in_axes=(1,1)), in_axes=(0,0)))
        outer_diag_blocks = p_outer(v_mean, v_mean) # [L,M,D,D]
        outer_offdiag_blocks = p_outer(v_mean[:,:,1:], v_mean[:,:,:-1]) # [L,M-1,D,D]

        xi_1 = v_mean # [L,D,M]
        xi_2_diag_blocks = outer_diag_blocks + S_inv_diag_blocks # [L,M,D,D]
        xi_2_offdiag_blocks = outer_offdiag_blocks + S_inv_offdiag_blocks # [L,M-1,D,D]

        return xi_1, xi_2_diag_blocks, xi_2_offdiag_blocks

    def exp_to_params(
        self, 
        xi_1: f64['L D M'], 
        xi_2_diag_blocks: f64['L M D D'], 
        xi_2_offdiag_blocks: f64['L M-1 D D']
    ) -> tuple[f64['L D M'], f64['L M D*(D+1)/2'], f64['L M-1 D D']]:
        """
        Converts the expectation parameters to their corresponding unconstrained variational parameters.

        Note: m = xi_1, BTD(R) = BTD(chol(S))

        For block-banded Cholesky decomposition from the inverse, we use the algorithm in (Asif and Moura, 2005). 

        Input shapes: xi_1 [L,D,M], xi_2_diag_blocks [L,M,D,D], xi_2_offdiag_blocks [L,M-1,D,D]
        Output shapes: v_mean [L,D,M], unc_R_diag_blocks [L,M,D*(D+1)/2], unc_R_offdiag_blocks [L,M-1,D,D] 
        """

        # Compute outer product of mean
        p_outer = jit(vmap(vmap(jnp.outer, in_axes=(1,1)), in_axes=(0,0)))
        outer_diag_blocks = p_outer(xi_1, xi_1) # [L,M,D,D]
        outer_offdiag_blocks = p_outer(xi_1[:,:,1:], xi_1[:,:,:-1]) # [L,M-1,D,D]

        S_inv_diag_blocks = xi_2_diag_blocks - outer_diag_blocks # [L,M,D,D]
        S_inv_offdiag_blocks = xi_2_offdiag_blocks - outer_offdiag_blocks # [L,M-1,D,D]

        @partial(vmap, in_axes=(0,0))
        def compute_diag_blocks(S_i_inv_diag_blocks, S_i_inv_offdiag_blocks):
            """Computes diagonal blocks of constrained R."""

            diag_final = S_i_inv_diag_blocks[-1,:,:] # [D,D]
            R_i_diag_final = jnp.linalg.cholesky(inv(diag_final, jitter=self.jitter)) # [D,D]

            def diag_helper(prev, idx):
                S_i_inv_diag_current = S_i_inv_diag_blocks[idx,:,:] # [D,D]
                S_i_inv_diag_next = S_i_inv_diag_blocks[idx+1,:,:] # [D,D]
                S_i_inv_offdiag = S_i_inv_offdiag_blocks[idx,:,:] # [D,D]

                schur = S_i_inv_diag_current - quadratic(S_i_inv_offdiag, S_i_inv_diag_next, jitter=self.jitter)
                schur_inverse = inv(schur, jitter=self.jitter)
                R_i_diag_block = jnp.linalg.cholesky(schur_inverse + self.jitter * jnp.eye(self.D))

                return prev, R_i_diag_block

            idxs = jnp.arange(0, self.M-1) # [0,...,M-2]
            dummy_init = jnp.zeros((self.D,self.D)) # [D,D]
            _, R_i_diag_blocks = lax.scan(diag_helper, dummy_init, idxs) # [M-1,D,D]
            R_i_diag_blocks = jnp.concatenate([R_i_diag_blocks, R_i_diag_final[None,:,:]], axis=0) # [M,D,D]

            return R_i_diag_blocks

        @partial(vmap, in_axes=(0,0,0))
        def compute_offdiag_blocks(S_i_inv_diag_blocks, S_i_inv_offdiag_blocks, R_i_diag_blocks):
            """Computes the off-diagonal blocks of constrained R."""

            def offdiag_helper(prev, idx):
                S_i_inv_diag = S_i_inv_diag_blocks[idx+1,:,:] # [D,D]
                S_i_inv_offdiag = S_i_inv_offdiag_blocks[idx,:,:] # [D,D]
                R_i_diag = R_i_diag_blocks[idx,:,:] # [D,D]

                R_i_offdiag_block = -inv_product(S_i_inv_offdiag, S_i_inv_diag, jitter=self.jitter) @ R_i_diag # [D,D]

                return prev, R_i_offdiag_block

            idxs = jnp.arange(0, self.M-1) # [0,...,M-2]
            dummy_init = jnp.zeros((self.D,self.D)) # [D,D]
            _, R_i_offdiag_blocks = lax.scan(offdiag_helper, dummy_init, idxs) # [M-1,D,D]

            return R_i_offdiag_blocks

        R_diag_blocks = compute_diag_blocks(S_inv_diag_blocks, S_inv_offdiag_blocks) # [L,M,D,D]
        R_offdiag_blocks = compute_offdiag_blocks(S_inv_diag_blocks, S_inv_offdiag_blocks, R_diag_blocks) # [L,M-1,D,D]

        # Unconstrain R
        unc_R_diag_blocks, unc_R_offdiag_blocks = jit(vmap(unconstrain_R, in_axes=(0,0)))(R_diag_blocks,
                                                                                          R_offdiag_blocks)

        # Mean parameter
        v_mean = xi_1 # [L,D,M]

        return v_mean, unc_R_diag_blocks, unc_R_offdiag_blocks
    
    def transition(
        self, 
        Z1: f64['L D'], 
        cov_infty: f64['L D D'], 
        times: Union[f64['T'], f64['T 1']], 
        key: chex.PRNGKey
    ) -> f64['L D T']:
        """Sample subsequent Z's given Z1."""

        T = times.shape[0]
        time_diffs = times[1:] - times[:-1]
        
        # Compute the state-transition matrix
        F = jit(vmap(self.compute_F))(self.model_params['unc_lengthscales'])
        
        def transition_helper(Z_prev, current):
            subkey, time_diff = current
            
            # Compute mean
            psi_k = jit(vmap(compute_psi, in_axes=(None,0)))(time_diff, F) # [L,D,D]
            Z_k_mean = jnp.einsum('ijk,ik->ij', psi_k, Z_prev) # [L,D,D], [L,D] -> [L,D]

            # Compute transition covariance
            phi_k = jit(vmap(cov_steady, in_axes=(None,0,0)))(time_diff, F, cov_infty) # [L,D,D]
            
            # Compute Cholesky decomposition for phi_k for more efficient sampling
            chol_k = jit(vmap(jnp.linalg.cholesky))(phi_k) # [L,D,D]
            
            # Sample from distribution
            epsilon = tfd.Normal(loc=0., scale=1.).sample((self.L,self.D), seed=subkey)
            Z_k = Z_k_mean + jnp.einsum('ijk,ik->ij', chol_k, epsilon) # [L,D]
            
            return Z_k, Z_k
        
        subkeys = jr.split(key, T-1)
        _, Z = lax.scan(transition_helper, Z1, (subkeys, time_diffs)) # [T-1,L,D]
        Z = jnp.vstack((Z1[None,:],Z)) # [T,L,D]
        Z = jnp.transpose(Z, axes=[1,2,0]) # [L,D,T]
            
        return Z
    
    def emission(
        self, 
        Z: f64['L D T'], 
        key: chex.PRNGKey
    ) -> f64['P T']:
        """Samples Y given W and Z."""

        Y_mean = self.model_params['W'] @ Z[:,0,:] # [P,L] * [L,T] -> [P,T]
        
        _, subkey = jr.split(key)
        std_normal = jnp.sqrt(softplus(self.model_params['unc_var'], low=EPS)) * jr.normal(subkey, Y_mean.shape)
        Y = Y_mean + std_normal
        
        return Y
    
    def sample(
        self, 
        times: Union[f64['T'], f64['T 1']], 
        key: chex.PRNGKey
    ) -> tuple[f64['L D T'], f64['P T']]:
        """Generates samples from FMOGP."""

        # Sample initial state from steady-state covariance
        cov_infty = self.compute_cov_infty(self.model_params['unc_k_vars'], 
                                           self.model_params['unc_lengthscales']) # [L,D,D]
        chol_cov_infty = jit(vmap(jnp.linalg.cholesky))(cov_infty) # [L,D,D]

        _, subkey = jr.split(key)
        epsilon = tfd.Normal(loc=0., scale=1.).sample((self.L,self.D), seed=subkey) # [L,D]
        Z1 = jnp.einsum('ijk,ik->ij', chol_cov_infty, epsilon) # [L,D]
        
        # Transition
        Z = self.transition(Z1, cov_infty, times, key) # [L,D,T]
        
        # Emission
        Y = self.emission(Z, key) # [P,T]
        
        return Z, Y
    
    def compute_Lambda(
        self,
        ind_psi_i: f64['M-1 D D'], 
        ind_phi_i: f64['M-1 D D'], 
        cov_infty_i: f64['D D']
    ):
        """Computes the prior precision matrix blocks for the inducing points of each latent GP."""

        # Compute diagonal blocks
        T1 = jit(vmap(inv, in_axes=(0,None)))(ind_phi_i, self.jitter) # [M-1,D,D]
        T2 = jit(vmap(quadratic, in_axes=(0,0)))(ind_psi_i, ind_phi_i) # [M-1,D,D]
        
        Lambda_diag_blocks = jnp.zeros((self.M,self.D,self.D)) # [M,D,D]
        Lambda_diag_blocks = Lambda_diag_blocks.at[0].add(inv(cov_infty_i, jitter=self.jitter))
        Lambda_diag_blocks = Lambda_diag_blocks.at[1:].add(T1)
        Lambda_diag_blocks = Lambda_diag_blocks.at[:-1].add(T2)
        
        # Compute lower off-diagonal blocks
        Lambda_offdiag_blocks = -jit(vmap(inv_product, in_axes=(0,0)))(ind_psi_i, ind_phi_i) # [M-1,D,D]
        
        return Lambda_diag_blocks, Lambda_offdiag_blocks
    
    def tridiag_inv(
        self, 
        diag_blocks: f64['M D D'], 
        offdiag_blocks: f64['M-1 D D'], 
        cholesky: Optional[bool] = False
    ):
        """
        Computes the diagonal and off-diagonal blocks of the inverse of a block-tridiagonal matrix.

        Implements the block-tridiagonal inversion algorithm in (Reuter and Hill, 2012).
        
        Shapes: diag_blocks [M,D,D], offdiag_blocks [M-1,D,D]
        """

        blocks = (diag_blocks, offdiag_blocks)
        
        # Compute Schur complements
        def schur_helper_A(A_prev, k, blocks=blocks):
            diag_blocks, offdiag_blocks = blocks
            
            chol_f = lambda: quadratic(offdiag_blocks[k], diag_blocks[k+1] - A_prev, jitter=self.jitter)
            inv_f = lambda: offdiag_blocks[k].T @ inv(diag_blocks[k+1] - A_prev, jitter=self.jitter) @ \
                            offdiag_blocks[k]

            A_k = lax.cond(cholesky, chol_f, inv_f)
            
            return A_k, A_k
        
        def schur_helper_B(B_prev, k, blocks=blocks):
            diag_blocks, offdiag_blocks = blocks

            chol_f = lambda: quadratic(offdiag_blocks[k-1].T, diag_blocks[k-1] - B_prev, jitter=self.jitter)
            inv_f = lambda: offdiag_blocks[k-1] @ inv(diag_blocks[k-1] - B_prev, jitter=self.jitter) @ \
                            offdiag_blocks[k-1].T
            
            B_k = lax.cond(cholesky, chol_f, inv_f)
            
            return B_k, B_k
        
        @partial(vmap, in_axes=(0,0,0,0))
        def offdiag_helper(diag_block, A, offdiag_block, inv_diag_block):
            #result = -jnp.linalg.inv(diag_block - A + self.jitter * jnp.eye(D)) @ offdiag_block @ inv_diag_block
            result = -inv(diag_block - A, jitter=self.jitter) @ offdiag_block @ inv_diag_block
            
            return result
        
        A_T = jnp.zeros((self.D,self.D))
        _, A = lax.scan(schur_helper_A, A_T, jnp.arange(0,self.M-1)[::-1]) # A = [M-1,D,D], index starts at M-2
        A = jnp.flip(A, axis=0) # Order from A_1, ...., A_{M-1}
        A = jnp.concatenate((A, A_T[None,:]), axis=0) # [M,D,D]
        
        B_1 = jnp.zeros((self.D,self.D))
        _, B = lax.scan(schur_helper_B, B_1, jnp.arange(1,self.M)) # B = [M-1,D,D], index starts at 1 
        B = jnp.concatenate((B_1[None,:], B), axis=0) # [M,D,D]
        
        # Compute main diagonal blocks
        inv_diag_blocks = jit(vmap(inv, in_axes=(0,None)))(diag_blocks - A - B, self.jitter) # [M,D,D]
        
        # Compute lower off-diagonal blocks
        inv_offdiag_blocks = offdiag_helper(diag_blocks[1:], A[1:], offdiag_blocks, inv_diag_blocks[:-1]) # [M-1,D,D]

        return inv_diag_blocks, inv_offdiag_blocks
    
    def tridiag_logdet(
        self, 
        diag_blocks: f64['M D D'], 
        offdiag_blocks: f64['M-1 D D'], 
        cholesky: Optional[bool] = False
    ) -> f64:
        """
        Computes the log-determinant of a block-tridiagonal matrix.

        Implements the log-determinant algorithm in (Salkuyeh, 2006).

        Shapes: diag_blocks [M,D,D], offdiag_blocks [M-1,D,D]
        """

        blocks = (diag_blocks, offdiag_blocks)
        
        def det_helper(A_prev, k, blocks=blocks):
            diag_blocks, offdiag_blocks = blocks
            
            chol_f = lambda: diag_blocks[k] - quadratic(offdiag_blocks[k-1].T, A_prev, jitter=self.jitter)
            inv_f = lambda: diag_blocks[k] - offdiag_blocks[k-1] @ inv(A_prev, jitter=self.jitter) @ \
                            offdiag_blocks[k-1].T

            A_k = lax.cond(cholesky, chol_f, inv_f)
            logdet_k = jnp.multiply(*jnp.linalg.slogdet(A_k))
            
            return A_k, logdet_k
            
        _, logdets = lax.scan(det_helper, diag_blocks[0], jnp.arange(1,self.M))
        logdet = jnp.multiply(*jnp.linalg.slogdet(diag_blocks[0])) + jnp.sum(logdets)
            
        return logdet
    
    @partial(vmap, in_axes=(None,0,0,0,0,0,0,0))
    def KL(
        self,
        v_mean_i: f64['D M'], 
        Lambda_i_diag_blocks: f64['M D D'], 
        Lambda_i_offdiag_blocks: f64['M-1 D D'], 
        R_i_diag_blocks: f64['M D*(D+1)/2'], 
        R_i_offdiag_blocks: f64['M-1 D D'],
        S_i_inv_diag_blocks: f64['M D D'], 
        S_i_inv_offdiag_blocks: f64['M-1 D D']
    ) -> f64:
        """Computes KL(q||p) for inducing points."""

        # v_mean_i: [D,T], Lambda_i_diag_blocks: [M,D,D], Lambda_i_offdiag_blocks: [M-1,D,D]
        def m_Lambda_m(
            v_mean_i: f64['D T'], 
            Lambda_i_diag_blocks: f64['M D D'], 
            Lambda_i_offdiag_blocks: f64['M-1 D D']
        ) -> f64:
            
            @jit
            @partial(vmap, in_axes=(1,0,1))
            def helper(A, B, C):
                return jnp.squeeze(A.T @ B @ C)
            
            result = jnp.sum(helper(v_mean_i, Lambda_i_diag_blocks, v_mean_i))
            result += 2 * jnp.sum(helper(v_mean_i[:,1:], Lambda_i_offdiag_blocks, v_mean_i[:,:-1]))

            return result
        
        # Convert to triangular matrix
        #triangular = tfb.FillTriangular(upper=False)
        #R_i_diag_blocks = triangular.forward(R_i_diag_blocks) # [M,D,D]
        S_i_diag_blocks, S_i_offdiag_blocks = self.R_to_S(R_i_diag_blocks, R_i_offdiag_blocks)
        #p_diag = jit(vmap(jnp.diag))
        #logdet_S_i = 2 * jnp.sum(jnp.log(p_diag(R_i_diag_blocks)))
        
        #kl = 0.5 * logdet_S_i
        kl = 0.5 * self.tridiag_logdet(S_i_diag_blocks, S_i_offdiag_blocks)
        kl += -0.5 * self.tridiag_logdet(Lambda_i_diag_blocks, Lambda_i_offdiag_blocks)
        kl += -0.5 * self.M * self.D
        kl += 0.5 * m_Lambda_m(v_mean_i, Lambda_i_diag_blocks, Lambda_i_offdiag_blocks)
        kl += 0.5 * (jnp.sum(Lambda_i_diag_blocks * S_i_inv_diag_blocks) + \
                     jnp.sum(Lambda_i_offdiag_blocks * S_i_inv_offdiag_blocks))
        
        return kl

    def compute_ELBO(
        self, 
        model_params: Dict,
        v_params: Dict, 
        batch: "Dataset"
    ) -> f64:
        """Computes the ELBO given a batch of data."""

        # Compute steady-state covariance
        cov_infty = self.compute_cov_infty(model_params['unc_k_vars'], 
                                           model_params['unc_lengthscales']) # [L,D,D]

        # Constrain R
        R_diag_blocks, R_offdiag_blocks = jit(vmap(constrain_R, in_axes=(0,0)))(v_params['unc_R_diag_blocks'],
                                                                                v_params['unc_R_offdiag_blocks'])

        @partial(vmap, in_axes=(1,0))
        def trace_W_Cov(W_i: f64['P'], S_i_inv_diag_blocks: f64['M D D']) -> f64:
            return jnp.linalg.norm(W_i)**2 * jnp.sum(S_i_inv_diag_blocks[:,0,0])
        
        ind_time_diffs = self.ind_times[1:] - self.ind_times[:-1]
        
        # Compute state-transition matrix
        F = jit(vmap(self.compute_F))(model_params['unc_lengthscales'])
        
        # Compute psi and phi
        ind_prior_psi = jit(vmap(vmap(compute_psi, in_axes=(0,None)), in_axes=(None,0)))(ind_time_diffs, F) # [L,M-1,D,D]
        ind_prior_phi = jit(vmap(vmap(cov_steady, in_axes=(0,None,None)),
                                 in_axes=(None,0,0)))(ind_time_diffs, F, cov_infty) # [L,M-1,D,D]

        # Computes the precision and covariance blocks for all latent SDEs
        # Note: Lambda_diag_blocks: [L,M,D,D], Lambda_offdiag_blocks: [L,M-1,D,D]
        Lambda_diag_blocks, Lambda_offdiag_blocks = jit(vmap(self.compute_Lambda, in_axes=(0,0,0)))(ind_prior_psi, 
                                                                                                    ind_prior_phi, 
                                                                                                    cov_infty)
        
        # Compute S from R
        # Note: S_diag_blocks [L,M,D,D], S_offdiag_blocks [L,M-1,D,D]
        S_diag_blocks, S_offdiag_blocks = jit(vmap(self.R_to_S, in_axes=(0,0)))(R_diag_blocks, R_offdiag_blocks)
        
        # Compute inverse blocks of S
        # Note: S_inv_diag_blocks [L,M,D,D], S_offdiag_blocks [L,M-1,D,D]
        S_inv_diag_blocks, S_inv_offdiag_blocks = jit(vmap(self.tridiag_inv, in_axes=(0,0)))(S_diag_blocks, S_offdiag_blocks)
        
        # Get predictions
        approx_mean, approx_cov = self.predict(batch.times, 
                                               F,
                                               model_params['unc_var'], 
                                               model_params['W'], 
                                               cov_infty, 
                                               v_params['v_mean'],
                                               S_inv_diag_blocks, 
                                               S_inv_offdiag_blocks, 
                                               output_latent=True)
        
        Y_hat = model_params['W'] @ approx_mean[:,:,0] # [P,L] x [L,V] -> [P,V]
        var = softplus(model_params['unc_var'], low=EPS)
        
        # Expected log-likelihood
        exp_loglik = -0.5 * batch.T * self.P * jnp.log(2 * jnp.pi * var)
        exp_loglik += -0.5 * (1./var) * jnp.linalg.norm(batch.Y - Y_hat)**2
        exp_loglik += -0.5 * (1./var) * jnp.sum(trace_W_Cov(model_params['W'], S_inv_diag_blocks))
        exp_loglik *= (self.T / batch.T) 

        # KL terms
        kl = jnp.sum(self.KL(v_params['v_mean'], Lambda_diag_blocks, Lambda_offdiag_blocks,
                             R_diag_blocks, R_offdiag_blocks, S_inv_diag_blocks, S_inv_offdiag_blocks))

        elbo = exp_loglik - kl

        return (elbo, exp_loglik, kl)
    
    def loss(
        self, 
        model_params: Dict,
        v_params: Dict,
        batch: "Dataset"
    ) -> f64:
        """Compute the negative ELBO given a batch of data."""

        elbo, exp_loglik, kl = self.compute_ELBO(model_params, v_params, batch)

        return -elbo, (exp_loglik, kl)

    def fit(
        self, 
        train_dataset: "Dataset", 
        n_steps: int, 
        batch_size: int, 
        lr: float, 
        lr_steps: int, 
        key: Optional[chex.PRNGKey] = jr.PRNGKey(0),
        log_rate: Optional[int] = 1,
        use_natgrad: Optional[bool] = True
    ) -> list[f64]:
        """Fits the FMOGP with mini-batch SGD and/or NGD."""

        # Initialize optimizers
        model_optimizer = optax.adam(lr['model_lr']) 
        opt_state = model_optimizer.init(self.model_params)

        if use_natgrad:
            # Log-linear increase scheduler
            scheduler = loglinear_schedule(init_value=lr['var_lr_init'], 
                                           end_value=lr['var_lr_end'], 
                                           transition_steps=lr_steps)

            v_optimizer = optax.chain(optax.scale_by_schedule(scheduler),
                                      optax.scale(-1.0))
        else:
            v_optimizer = optax.adam(lr['var_adam_lr'])

        v_opt_state = v_optimizer.init(self.v_params)

        # Collect initial setting
        init_state = (self.model_params, self.v_params, opt_state, v_opt_state)

        # Track total training time
        start_time = datetime.now()

        # Single optimization step
        @progress_bar_scan2(n_steps, log_rate)
        def step(carry, current):
            # Unpack
            model_params, v_params, opt_state, v_opt_state = carry
            step_idx, key = current

            # Get random mini-batch
            batch = get_batch(train_dataset, batch_size, key)

            # Compute gradients
            out, grads = jit(value_and_grad(self.loss, argnums=(0,1), has_aux=True))(model_params, 
                                                                                     v_params, 
                                                                                     batch)
            elbo = -out[0]
            exp_loglik = out[1][0]
            kl = out[1][1]

            # Update model parameters via ADAM
            p_clip_grads = Partial(clip_grads, max_norm=1e4)
            clipped_grads = tree_map(p_clip_grads, grads[0])

            model_updates, opt_state = model_optimizer.update(clipped_grads, opt_state)
            model_params = optax.apply_updates(model_params, model_updates)

            # Update variational parameters via NGD
            def natgrad_update(v_params, v_opt_state):
                v_grads = (grads[1]['v_mean'].astype(jnp.float64), 
                           grads[1]['unc_R_diag_blocks'].astype(jnp.float64),
                           grads[1]['unc_R_offdiag_blocks'].astype(jnp.float64))

                # Compute gradient wrt expectation parameters
                exp_params = self.params_to_exp(v_params['v_mean'].astype(jnp.float64),
                                                v_params['unc_R_diag_blocks'].astype(jnp.float64),
                                                v_params['unc_R_offdiag_blocks'].astype(jnp.float64))

                _, exp_vjp = vjp(self.exp_to_params, *exp_params)
                grads_exp = exp_vjp(v_grads)

                # Compute natural gradient wrt parameters
                nat_params = self.params_to_nat(v_params['v_mean'].astype(jnp.float64),
                                                v_params['unc_R_diag_blocks'].astype(jnp.float64),
                                                v_params['unc_R_offdiag_blocks'].astype(jnp.float64))

                _, natgrads = jvp(self.nat_to_params, nat_params, grads_exp)

                p_clip_grads = Partial(clip_grads, max_norm=1e4)
                clipped_natgrads = tree_map(p_clip_grads, natgrads)

                natgrad_dict = dict(v_mean=clipped_natgrads[0],
                                    unc_R_diag_blocks=clipped_natgrads[1],
                                    unc_R_offdiag_blocks=clipped_natgrads[2])

                v_updates, v_opt_state = v_optimizer.update(natgrad_dict, v_opt_state)
                v_params = optax.apply_updates(v_params, v_updates)

                return v_params, v_opt_state

            # Update variational parameters via Adam
            def adam_update(v_params, v_opt_state):
                p_clip_grads = Partial(clip_grads, max_norm=1e4)
                clipped_v_grads = tree_map(p_clip_grads, grads[1])

                v_updates, v_opt_state = v_optimizer.update(clipped_v_grads, v_opt_state)
                v_params = optax.apply_updates(v_params, v_updates)

                return v_params, v_opt_state

            v_params, v_opt_state = lax.cond(use_natgrad, 
                                             natgrad_update, 
                                             adam_update, 
                                             v_params,
                                             v_opt_state)

            # Collect updates
            carry = (model_params, v_params, opt_state, v_opt_state)

            return carry, (elbo, exp_loglik, kl)

        # Optimize the model
        step_idxs = jnp.arange(0, n_steps)
        keys = jr.split(key, n_steps)
        (model_params, v_params, _, _), metrics = lax.scan(step, init_state, (step_idxs, keys))

        train_time = datetime.now() - start_time
        print(f'[Total training time] {str(train_time).split(".")[0]}')

        # Unpack parameters
        self.model_params = model_params
        self.v_params = v_params

        return metrics

    def past_moments(
        self, 
        args: Dict
    ) -> tuple[f64['D'], f64['D D']]:
        """Computes the approximate moments for past time points."""

        # Unpack arguments
        past_time = args['time']
        cov_infty_i = args['cov_infty_i']
        F_i = args['F_i']
        v_mean_i = args['v_mean_i']
        S_i_inv_diag_blocks = args['S_i_inv_diag_blocks']

        # Compute the time difference
        time_diff = self.first_ind_time - past_time

        psi = compute_psi(time_diff, F_i)
        phi = cov_steady(time_diff, F_i, cov_infty_i)

        chol_infty_i = jnp.linalg.cholesky(cov_infty_i + self.jitter * jnp.eye(self.D)) # [D,D]
        chol_factor_i = jscipy.linalg.solve_triangular(chol_infty_i, psi @ cov_infty_i, lower=True) # [D,D]
        factor_i = jscipy.linalg.solve_triangular(chol_infty_i.T, chol_factor_i, lower=False) # [D,D]

        # Variational parameters of the first inducing point
        v_mean_i_first = lax.dynamic_slice(v_mean_i, (0,0), (self.D,1)) # [D,1]
        S_i_inv_diag = lax.dynamic_slice(S_i_inv_diag_blocks, (0,0,0), (1,self.D,self.D)).squeeze() # [D,D]

        # Compute approximate mean and covariance
        approx_mean = (factor_i.T @ v_mean_i_first).squeeze() # [D,1] -> [D,]
        approx_cov = factor_i.T @ S_i_inv_diag @ factor_i + cov_infty_i - chol_factor_i.T @ chol_factor_i # [D,D]

        return approx_mean, approx_cov

    def smooth_moments(
        self, 
        args: Dict
    ) -> tuple[f64['D'], f64['D D']]:
        """Computes the approximate moments for in-between time points."""

        # Unpack arguments
        smooth_time = args['time']
        cov_infty_i = args['cov_infty_i']
        F_i = args['F_i']
        v_mean_i = args['v_mean_i']
        S_i_inv_diag_blocks = args['S_i_inv_diag_blocks']
        S_i_inv_offdiag_blocks = args['S_i_inv_offdiag_blocks']

        # Find closest two inducing points
        lower_idx = (jnp.where(self.ind_times < smooth_time, 1, 0).argmin() - 1).astype(jnp.int64)
        upper_idx = (jnp.where(self.ind_times > smooth_time, 1, 0).argmax()).astype(jnp.int64)

        lower = lax.dynamic_slice(self.ind_times, (lower_idx,), (1,)).squeeze()
        upper = lax.dynamic_slice(self.ind_times, (upper_idx,), (1,)).squeeze()

        v_mean_i_lower = lax.dynamic_slice(v_mean_i, (0,lower_idx), (self.D,1)) # [D,1]
        v_mean_i_upper = lax.dynamic_slice(v_mean_i, (0,upper_idx), (self.D,1)) # [D,1]
        v_mean_i_pair = jnp.vstack((v_mean_i_lower, v_mean_i_upper)) # [2*D,1]

        S_i_inv_diag_pair = lax.dynamic_slice(S_i_inv_diag_blocks, (lower_idx,0,0), (2,self.D,self.D)).squeeze() # [2,D,D]
        S_i_inv_offdiag = lax.dynamic_slice(S_i_inv_offdiag_blocks, (lower_idx,0,0), (1,self.D,self.D)).squeeze() # [D,D]
        S_i_inv_pair = jscipy.linalg.block_diag(*S_i_inv_diag_pair) # [2*D,2*D]
        S_i_inv_pair = S_i_inv_pair.at[self.D:,:self.D].add(S_i_inv_offdiag) # [2*D,2*D]
        S_i_inv_pair = S_i_inv_pair.at[:self.D,self.D:].add(S_i_inv_offdiag.T) # [2*D,2*D]

        # Compute time differences
        upper_time_diff = upper - smooth_time # t_{k+1} - t_*
        lower_time_diff = smooth_time - lower # t_* - t_k

        # Compute required terms
        Sigma_lower = compute_psi(lower_time_diff, F_i) @ cov_infty_i # \bmSig_{*,k}, [D,D]
        Sigma_upper = cov_infty_i @ compute_psi(upper_time_diff, F_i).T # \bmSig_{*,k+1}, [D,D]
        
        phi = cov_steady(upper - lower, F_i, cov_infty_i) # \bmPhi_{k+1}, [D,D]
        psi = compute_psi(upper - lower, F_i) # \bmPsi_{k+1}, [D,D]
        
        phi_inv_psi_i = inv_product(psi, phi, jitter=self.jitter) # \bmPhi_{k+1}^{-1} \bmPsi_{k+1}, [D,D]
        psi_phi_inv_psi_i = quadratic(psi, phi, jitter=self.jitter) # \bmPsi_{k+1}^T \bmPhi_{k+1}^{-1} \bmPsi_{k+1}, [D,D]

        # (\bmSig_{*,k+1} - \bmSig_{*,k} @ \bmPsi_{k+1}.T) @ \bmPhi_{k+1}^{-1}
        upper_to_time = (Sigma_upper - Sigma_lower @ psi.T) @ inv(phi, jitter=self.jitter) # [D,D]
        lower_to_time = compute_psi(lower_time_diff, F_i) - upper_to_time @ psi # [D,D]
        mean_transform = jnp.hstack((lower_to_time, upper_to_time)) # [D,2*D]

        # Compute approximate mean
        approx_mean = (mean_transform @ v_mean_i_pair).squeeze() # [D,1] -> [D,]

        # Compute prior conditional covariance
        cov_prior = cov_steady(lower_time_diff, F_i, cov_infty_i) - \
                    quadratic(Sigma_upper.T, phi, jitter=self.jitter) + \
                    Sigma_upper @ phi_inv_psi_i @ Sigma_lower.T - \
                    Sigma_lower @ psi_phi_inv_psi_i @ Sigma_lower.T + \
                    Sigma_lower @ phi_inv_psi_i.T @ Sigma_upper.T

        # Compute approximate covariance
        approx_cov = cov_prior + mean_transform @ S_i_inv_pair @ mean_transform.T # [D,D]

        return approx_mean, approx_cov

    def forecast_moments(
        self, 
        args: Dict
    ) -> tuple[f64['D'], f64['D D']]:
        """Computes the approximate moments for future time points."""

        # Unpack arguments
        forecast_time = args['time']
        cov_infty_i = args['cov_infty_i']
        F_i = args['F_i']
        v_mean_i = args['v_mean_i']
        S_i_inv_diag_blocks = args['S_i_inv_diag_blocks']

        # Compute the time difference
        time_diff = forecast_time - self.last_ind_time

        psi = compute_psi(time_diff, F_i)
        phi = cov_steady(time_diff, F_i, cov_infty_i)

        # Variational parameters for last inducing point
        v_mean_i_final = lax.dynamic_slice(v_mean_i, (0,-1), (self.D,1)) # [D,1]
        S_i_inv_diag = lax.dynamic_slice(S_i_inv_diag_blocks, (-1,0,0), (1,self.D,self.D)).squeeze() # [D,D]

        # Compute approximate mean and covariance
        approx_mean = (psi @ v_mean_i_final).squeeze() # [D,1] -> [D,]
        approx_cov = psi @ S_i_inv_diag @ psi.T + phi # [D,D]

        return approx_mean, approx_cov

    @partial(vmap, in_axes=(None,None,None,0,0,0,0,0))
    @partial(vmap, in_axes=(None,0,0,None,None,None,None,None))
    def approx_moments(
        self, 
        time: f64, 
        mode: i64, 
        cov_infty_i: f64['D D'], 
        F_i: f64['D D'], 
        v_mean_i: f64['D M'], 
        S_i_inv_diag_blocks: f64['M D D'], 
        S_i_inv_offdiag_blocks: f64['M-1 D D']
    ) -> tuple[f64['D'], f64['D D']]:
        """Computes the approximate moments for a given time point."""

        # Gather arguments
        args = dict(time=time,
                    cov_infty_i=cov_infty_i,
                    F_i=F_i,
                    v_mean_i=v_mean_i,
                    S_i_inv_diag_blocks=S_i_inv_diag_blocks,
                    S_i_inv_offdiag_blocks=S_i_inv_offdiag_blocks)

        # Call appropriate approximation call
        return lax.switch(mode, [self.past_moments, self.smooth_moments, self.forecast_moments], args)

    def predict(
        self, 
        times: Union[f64['V'], f64['V 1']], 
        F: f64['L D D'],  
        unc_var: f64, 
        W: f64['P L'], 
        cov_infty: f64['L D D'], 
        v_mean: f64['L D M'], 
        S_inv_diag_blocks: f64['L M D D'], 
        S_inv_offdiag_blocks: f64['L M-1 D D'], 
        full_result: Optional[bool] = False, 
        output_latent: Optional[bool] = False
    ):
        """
        Generate predictions for the given time points.
        
        Note: The output_latent flag makes the function return the approximate moments of the latent states.
        """

        times = times.squeeze()

        # Separate interpolation and extrapolation
        modes_1 = jnp.where(times < self.first_ind_time, 0, 1)
        modes_2 = jnp.where(times > self.last_ind_time, 1, 0)
        modes = modes_1 + modes_2

        # Compute approximate mean and covariance
        args = [times, modes, cov_infty, F, v_mean, S_inv_diag_blocks, S_inv_offdiag_blocks]
        approx_mean, approx_cov = jit(self.approx_moments)(*args)

        if output_latent:
            return approx_mean, approx_cov

        # Compute predicted mean
        Y_hat = W @ approx_mean[:,:,0] # [P,L] x [L,V] -> [P,V]

        def compute_pred_cov(approx_cov_t):
            """
            Computes the predictive variance.
            
            Note: approx_cov_t [L,D,D]
            """

            var = softplus(unc_var, low=EPS)
            scale = approx_cov_t[:,0,0] # [L,]
            scale_mat = jnp.diag(scale)
            pred_cov = W @ scale_mat @ W.T + var * jnp.eye(self.P) # [P,P]

            return pred_cov

        Y_cov = vmap(compute_pred_cov, in_axes=1)(approx_cov) # [V,P,P]

        if full_result:
            prediction = {'approx_mean': approx_mean, 'approx_cov': approx_cov,
                          'Y_hat': Y_hat, 'Y_cov': Y_cov}

            return prediction

        else:
            return Y_hat, Y_cov

    def predict_ind(self) -> list[f64['P M'], f64['M P P']]:
        """Computes the approximate mean and covariance at the inducing time points."""

        Y_hat = self.model_params['W'] @ self.v_params['v_mean'][:,0,:] # [P,M]

        def compute_pred_cov(approx_cov_t):
            """
            Computes the predictive variance.
            
            Note: approx_cov_t [L,D,D]
            """

            var = softplus(self.model_params['unc_var'], low=EPS)
            W = self.model_params['W']
            scale = approx_cov_t[:,0,0] # [L,]
            scale_mat = jnp.diag(scale)
            pred_cov = W @ scale_mat @ W.T + var * jnp.eye(self.P) # [P,P]

            return pred_cov

        # Constrain R
        R_diag_blocks, R_offdiag_blocks = jit(vmap(constrain_R, in_axes=(0,0)))(self.v_params['unc_R_diag_blocks'],
                                                                                self.v_params['unc_R_offdiag_blocks'])

        # Compute S from R
        # Note: S_diag_blocks [L,M,D,D], S_offdiag_blocks [L,M-1,D,D]
        S_diag_blocks, S_offdiag_blocks = jit(vmap(self.R_to_S, in_axes=(0,0)))(R_diag_blocks, 
                                                                                R_offdiag_blocks)
        
        # Compute inverse blocks of S
        # Note: S_inv_diag_blocks [L,M,D,D], S_offdiag_blocks [L,M-1,D,D]
        S_inv_diag_blocks, S_inv_offdiag_blocks = jit(vmap(self.tridiag_inv, in_axes=(0,0)))(S_diag_blocks, 
                                                                                             S_offdiag_blocks)

        Y_cov = jit(vmap(compute_pred_cov, in_axes=1))(S_inv_diag_blocks) # [M,P,P]

        return Y_hat, Y_cov