# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools

import jax
import jax.numpy as jnp

from ferminet_ecp.integral import special
from ferminet_ecp.integral.quadrature import Quadrature


def numerical_integral_exact(psi, r_atom, walkers, ls, quadrature: Quadrature):
    '''
    ref: Nonlocal pseudopotentials and diffusion Monte Carlo, equation 28

    inputs:
        psi: wave function that psi(walkers) returns a complex number
        r_atom: shape (3,)
        walkers: shape (n_electron, 3)
        ls: shape(l_number,) values of l to evaluate
        quadrature: A quadrature object to do numerical integration.
    returns:
        value of the integral \int (2l+1) * P_l(cos theta) psi(r1,..,ri,..)
        shape (n_electron, l_number)
    '''

    n_electron = walkers.shape[0]
    ri = jnp.linalg.norm(walkers-r_atom, axis=-1)
    res = jnp.zeros((n_electron, len(ls)))
    normal_walkers = (walkers-r_atom) / ri[:, None]
    psi_vec = jax.vmap(psi, in_axes=0)

    for j, l in enumerate(ls):
        Pl_ = lambda x: special.legendre(x, l)

        def Pl(i, x):
            tmp = Pl_(jnp.matmul(x, normal_walkers[i, :]))
            return tmp
            # (ro,np)

        def psi_r(i, x):
            coords = x.reshape(-1, 3) * ri[i] + r_atom
            new_walkers = jnp.tile(walkers, (coords.shape[0],) + (1, 1))
            new_walkers = jax.ops.index_update(new_walkers, jax.ops.index[:, i, :], coords)
            res = psi_vec(new_walkers)
            return res.reshape(x.shape[:-1])

        def product(i, x):
            return Pl(i, x) * psi_r(i, x)

        def integral(i, res):
            result = quadrature(lambda x: product(i, x)) * (2 * l + 1)
            res = jax.ops.index_update(res, jax.ops.index[i, j], result)
            return res

        res = jax.lax.fori_loop(0, n_electron, integral, res)


    return res
