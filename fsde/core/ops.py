import sys
sys.path.append('../../') # Add FSDE root

import jax
jax.config.update("jax_enable_x64", True) # For setting default dtype to float64
import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap, pmap, lax, custom_vjp
from jax.tree_util import Partial
import jax.scipy as jscipy
from jax.scipy.linalg import expm
from jax.experimental.ode import odeint
import os
import numpy as np
import copy
import scipy
from scipy.linalg import solve_continuous_lyapunov

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

# Computes the blocks of a block tridiagonal matrix given the diagonal and off-diagonal blocks of the Cholesky factor
# Note: Assumes R_diag_blocks is not in triangular block form
def tridiag_blocks_from_chol(R_diag_blocks, R_offdiag_blocks):
    # Convert diagonal blocks to lower triangular blocks
    triangular = tfb.FillTriangular(upper=False)
    R_diag_blocks = vmap(triangular.forward, in_axes=(0,), out_axes=0)(R_diag_blocks) # [L,T,D,D]

    # Compute the diagonal blocks
    p_einsum = Partial(jnp.einsum, 'ijk,ilk->ijl')
    S_diag_blocks_1 = vmap(p_einsum, in_axes=(0,0), out_axes=0)(R_diag_blocks, R_diag_blocks) # [L,T,D,D]
    S_diag_blocks_2 = vmap(p_einsum, in_axes=(0,0), out_axes=0)(R_offdiag_blocks, R_offdiag_blocks) # [L,T-1,D,D]
    S_diag_blocks = S_diag_blocks_1
    S_diag_blocks = S_diag_blocks.at[:,1:,:,:].add(S_diag_blocks_2) # [L,T,D,D]

    # Compute the off-diagonal blocks
    S_offdiag_blocks = vmap(p_einsum, in_axes=(0,0), out_axes=0)(R_offdiag_blocks, R_diag_blocks[:,:-1,:,:]) # [L,T-1,D,D]

    return S_diag_blocks, S_offdiag_blocks

# Constructs a block tridiagonal matrix given the diagonal and off-diagonal blocks
# NOTE: diag_blocks [T,D,D], offdiag_blocks [T-1,D,D]
def construct_tridiag(diag_blocks, offdiag_blocks, compact=True):
    if compact:
        triangular = tfb.FillTriangular(upper=False)
        diag_blocks = triangular.forward(diag_blocks)
    
    T, D = diag_blocks.shape[:-1]
    diag = jscipy.linalg.block_diag(*diag_blocks)
    offdiag = jscipy.linalg.block_diag(*offdiag_blocks)

    full = jnp.copy(diag)
    full = full.at[D:,0:(T-1)*D].add(offdiag) # Add lower blocks
    full = full.at[0:(T-1)*D,D:].add(offdiag.T) # Add upper blocks

    return full

# Constructs a block bidiagonal matrix given the diagonal and off-diagonal blocks
# NOTE: diag_blocks [T,D,D], offdiag_blocks [T-1,D,D]
def construct_bidiag(diag_blocks, offdiag_blocks, compact=True):
    if compact:
        triangular = tfb.FillTriangular(upper=False)
        diag_blocks = triangular.forward(diag_blocks)
    
    T, D = diag_blocks.shape[:-1]
    diag = jscipy.linalg.block_diag(*diag_blocks)
    offdiag = jscipy.linalg.block_diag(*offdiag_blocks)

    full = jnp.copy(diag)
    full = full.at[D:,0:(T-1)*D].add(offdiag) # Add lower blocks
    
    return full

# Fills in the diagonal of a with val
def fill_diagonal(a, val):
    assert(a.ndim >= 2)
    i, j = jnp.diag_indices(min(a.shape[-2:]))
    
    return a.at[..., i, j].set(val)

# Wrapper around jnp.linalg.inv with jitter
def inv(A, jitter=1e-10):
    d = A.shape[0]
    result = jnp.linalg.inv(A + jitter * jnp.eye(d))

    return result

# Computes A^T B^{-1} A
def quadratic(A, B, jitter=1e-10, cholesky=False):
    d = B.shape[0]

    def chol_f():
        L = jscipy.linalg.cholesky(B + jitter * jnp.eye(d), lower=True)
        S = jscipy.linalg.solve_triangular(L, A, lower=True)
        result = S.T @ S

        return result

    def inv_f():
        result = A.T @ inv(B, jitter=jitter) @ A

        return result
    
    result = lax.cond(cholesky, chol_f, inv_f)
            
    return result
        
# Computes B^{-1} A
def inv_product(A, B, jitter=1e-10, cholesky=False):
    d = B.shape[0]

    def chol_f():
        L = jscipy.linalg.cholesky(B + jitter * jnp.eye(d), lower=True)
        S = jscipy.linalg.solve_triangular(L, A, lower=True) # L^T B^{-1} A
        result = jscipy.linalg.solve_triangular(L.T, S, lower=False)

        return result

    def inv_f():
        result = inv(B, jitter=jitter) @ A

        return result

    result = lax.cond(cholesky, chol_f, inv_f)
            
    return result

# Computes the Gamma function
def gamma(x):
    return jnp.exp(lax.lgamma(x))

# Computes the softplus function
def softplus(x, low=0.):
    #eps = jnp.finfo(jnp.float64).eps
    #eps = 0.
    #return jnp.log(1 + jnp.exp(x + eps))
    return jnp.maximum(0., x) + jnp.log(1. + jnp.exp(-jnp.abs(x))) + low

# Computes the inverse of the softplus function
def inv_softplus(x, low=0.):
    return jnp.log(jnp.exp(x - low) - 1)

# Computes the log probability of a Gaussian distribution
def compute_log_normal(x, mu, sigma):
    d = sigma.shape[0]
    log_prob = -0.5 * (d * jnp.log(2*jnp.pi) + jnp.linalg.slogdet(sigma)[-1] + \
                        jnp.squeeze(quadratic(x - mu, sigma)))
        
    return log_prob

# Computes the logdet of a matrix
def compute_logdet(A):
    return jnp.multiply(*jnp.linalg.slogdet(A))

# Finds the solution to a continuous-time Lyapunov equation
# Note: Custom VJP defined to allow reverse-mode differentiation
@custom_vjp
def solve_lyapunov(A, B):
    A = np.asarray(A)
    B = np.asarray(B)

    result = solve_continuous_lyapunov(A, B)

    return jnp.asarray(result)

def lyapunov_fwd(A, B):
    result = solve_lyapunov(A, B)

    return result, (result, A)

def lyapunov_bwd(res, g):
    res = [np.asarray(e) for e in res]
    lagrange = solve_continuous_lyapunov(res[1].T, -g)

    return (lagrange @ res[0].T + lagrange.T @ res[0], -lagrange)

solve_lyapunov.defvjp(lyapunov_fwd, lyapunov_bwd)

# Computes the MSE
def compute_MSE(A, B):
    assert(A.shape == B.shape)

    return (1./A.size) * jnp.sum((A - B)**2)

# Computes the MAE
def compute_MAE(A, B):
    assert(A.shape == B.shape)

    return (1./A.size) * jnp.sum(jnp.abs(A - B))