# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file may have been modified by Bytedance Inc. (“Bytedance Modifications”).
# All Bytedance Modifications are Copyright 2021 Bytedance Inc.

"""Evaluating the Hamiltonian on a wavefunction."""

from ferminet import networks
import jax
from jax import lax
import jax.numpy as jnp

from ferminet_ecp.integral import pseudoPotential
from ferminet_ecp.integral.quadrature import get_quadrature


def local_kinetic_energy(f):
    r"""Creates a function to for the local kinetic energy, -1/2 \nabla^2 ln|f|.

    Args:
      f: Callable with signature f(params, data), where params is the set of
        (model) parameters of the (wave)function and data is the configurations to
        evaluate f at, and returns the values of the log magnitude of the
        wavefunction at those configurations.

    Returns:
      Callable with signature lapl(params, data), which evaluates the local
      kinetic energy, -1/2f \nabla^2 f = -1/2 (\nabla^2 log|f| +
      (\nabla log|f|)^2).

    MODIFICATION FROM FERMINET: None
    """

    def _lapl_over_f(params, x):
        n = x.shape[0]
        eye = jnp.eye(n)
        grad_f = jax.grad(f, argnums=1)
        grad_f_closure = lambda y: grad_f(params, y)

        def _body_fun(i, val):
            primal, tangent = jax.jvp(grad_f_closure, (x,), (eye[i],))
            return val + primal[i] ** 2 + tangent[i]

        return -0.5 * lax.fori_loop(0, n, _body_fun, 0.0)

    return _lapl_over_f


def potential_energy(r_ae, r_ee, atoms, charges):
    """Returns the potential energy for this electron configuration.

    Args:
      r_ae: Shape (nelectrons, natoms). r_ae[i, j] gives the distance between
        electron i and atom j.
      r_ee: Shape (neletrons, nelectrons, :). r_ee[i,j,0] gives the distance
        between electrons i and j. Other elements in the final axes are not
        required.
      atoms: Shape (natoms, ndim). Positions of the atoms.
      charges: Shape (natoms). Nuclear charges of the atoms.

    MODIFICATION FROM FERMINET: None
    """
    v_ee = jnp.sum(jnp.triu(1 / r_ee[..., 0], k=1))
    v_ae = -jnp.sum(charges / r_ae[..., 0])  # pylint: disable=invalid-unary-operand-type
    r_aa = jnp.linalg.norm(atoms[None, ...] - atoms[:, None], axis=-1)
    v_aa = jnp.sum(
        jnp.triu((charges[None, ...] * charges[..., None]) / r_aa, k=1))
    return v_ee + v_ae + v_aa


def local_energy(f, atoms, charges):
    """Creates function to evaluate the local energy.

    Args:
      f: Callable with signature f(data, params) which returns the log magnitude
        of the wavefunction given parameters params and configurations data.
      atoms: Shape (natoms, ndim). Positions of the atoms.
      charges: Shape (natoms).Nuclear charges of the atoms.

    Returns:
      Callable with signature e_l(params, data) which evaluates the local energy
      of the wavefunction given the parameters params and a single MCMC
      configuration in data.

    MODIFICATION FROM FERMINET: None
    """
    ke = local_kinetic_energy(f)

    def _e_l(params, x):
        """Returns the total energy.

        Args:
          params: network parameters.
          x: MCMC configuration.
        """
        _, _, r_ae, r_ee = networks.construct_input_features(x, atoms)
        potential = potential_energy(r_ae, r_ee, atoms, charges)
        kinetic = ke(params, x)
        return potential + kinetic

    return _e_l


def ecp(pe, pa, ecp_coe):
    """
    read ecp coeffs from pyscf obj

    NEWLY ADDED
    """
    norm = jnp.linalg.norm(pe[:, None, :] - pa, axis=-1)
    res = []
    for _, l in ecp_coe:
        result = 0
        for power, coe in enumerate(l):
            for coeff in coe:
                result = result + norm[:, 0] ** (power - 2) * jnp.exp(- coeff[0] * norm[:, 0] ** 2) * \
                         coeff[1]
        res.append(result)
    res = jnp.stack(res, axis=-1)
    return res


def non_local_energy(fs, pyscf_mole, ecp_quadrature_id=None):
    """
    Calculate Ecp energy.

    NEWLY ADDED
    """
    quadrature = get_quadrature(ecp_quadrature_id)

    def psi(params, x):

        sign_and_log = fs(params, x)

        return jnp.exp(sign_and_log[1]) * sign_and_log[0]

    def non_local(pe, pa, psi, l_list):
        res = pseudoPotential.numerical_integral_exact(psi, pa, pe, l_list, quadrature)
        return res / (4 * jnp.pi * psi(pe))

    def non_local_sum(params, x):
        res = 0
        pe = x.reshape(-1, 3)
        for sym, coord in pyscf_mole._atom:
            result = 0
            if sym in pyscf_mole._ecp:
                pa = jnp.array(coord)
                ecp_coe = pyscf_mole._ecp[sym][1]
                l_list = list(range(len(ecp_coe) - 1))
                ecp_list = ecp(pe, pa, ecp_coe)
                result = (jnp.sum(ecp_list[..., 1:] * non_local(pe, pa, lambda x: psi(params, x.flatten()), l_list),
                                  axis=-1)
                          + ecp_list[..., 0])
            res = res + result
        return jnp.sum(res, axis=-1)

    return non_local_sum
