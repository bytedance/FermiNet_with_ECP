# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import time

import jax.numpy as jnp
import jax

INTEGRAL_SAMPLE_SIZE = 1
DEFAULT_QUADRATURE_ID = 'icosahedron_12'


def get_quadrature(quadrature_id):
    """
    Args:
        quadrature_id: Expected to be "quadrature_type" + "_" + "number of points
        used in quadrature". It could also be None, in which case a default one
        will be used.
    """
    quadrature_id = quadrature_id or DEFAULT_QUADRATURE_ID

    ALL_QUADRATURES = {
        'octahedron_26': Octahedron(26),
        'icosahedron_12': Icosahedron(12)
    }
    return ALL_QUADRATURES[quadrature_id]


def expand_sign(l):
    """
     expand the set with signs,
     example: [a,b,c] => [+/- a, +/- b, +/- c]

    """
    if len(l) == 1:
        res = [[l[0]]]
        if l[0] != 0:
            res = res + [[-l[0]]]
        return res
    rests = expand_sign(l[1:])

    res = [[l[0]] + s for s in rests]
    if l[0] != 0:
        res += [[-l[0]] + s for s in rests]
    return res


class Quadrature():
    def __init__(self, n_p):
        self.np = n_p

    def integrate(self, f, rotationM):

        pts = jnp.einsum('ijk,kl->ijl', rotationM, self.pts.T)
        pts = pts.transpose(0, 2, 1)
        evl = f(pts)
        nums = jnp.sum(evl * self.coefs, axis=-1)
        return jnp.mean(nums, axis=-1)

    @staticmethod
    def sample_orientation(N):
        # sample z orientation
        if (N == 0):
            return jnp.eye(3)[None, ...]
        seed = int(1e6 * time.time())
        key = jax.random.PRNGKey(seed)
        phi_key, theta_key = jax.random.split(key)

        phi = jax.random.uniform(phi_key, shape=(N,)) * jnp.pi * 2
        costheta = 1.0 - 2 * jax.random.uniform(theta_key, shape=(N,))
        sintheta = jnp.sqrt(1.0 - costheta ** 2)

        sinphi = jnp.sin(phi)
        cosphi = jnp.cos(phi)
        sinphi2 = sinphi ** 2
        cosphi2 = cosphi ** 2

        M11 = sinphi2 + costheta * cosphi2
        M12 = sinphi * cosphi * (costheta - 1)
        M13 = sintheta * cosphi

        M21 = M12
        M22 = cosphi2 + costheta * sinphi2
        M23 = sintheta * sinphi

        M31 = -M13
        M32 = - M23
        M33 = costheta

        M = jnp.vstack([M11, M12, M13, M21, M22, M23, M31, M32, M33]).T
        M = M.reshape(-1, 3, 3)

        return M

    def __call__(self, f, N=INTEGRAL_SAMPLE_SIZE):
        '''
        /int f(x_1,x_2,x_3) dOmega over the sphere

        :param f: cartisian coords [3,]
        :param N:
        :return:[1,]
        '''
        Ms = self.sample_orientation(N)

        res = self.integrate(f, Ms) * jnp.pi * 4.0

        return res


class Octahedron(Quadrature):
    def __init__(self, n_p):
        super(Octahedron, self).__init__(n_p)

        A_num = 6
        B_num = 12
        C_num = 8
        D_num = 24

        self.coefs = {
            6: jnp.array([1. / 6.] * A_num),
            18: jnp.array([1. / 30.] * A_num + [1. / 15.] * B_num),
            26: jnp.array([1. / 21.] * A_num + [4. / 105.] * B_num + [27. / 840.] * C_num),
            50: jnp.array(
                [4. / 315.] * A_num + [64. / 2835.] * B_num + [27. / 1280.] * C_num + [14641. / 725760.] * D_num)
        }

        self.pts = expand_sign([1, 0, 0]) + expand_sign([0, 1, 0]) + expand_sign([0, 0, 1])
        p = 1. / jnp.sqrt(2.)
        self.pts += expand_sign([p, p, 0]) + expand_sign([p, 0, p]) + expand_sign([0, p, p])
        q = 1. / jnp.sqrt(3.)
        self.pts += expand_sign([q, q, q])
        r = 1. / jnp.sqrt(11.)
        s = 3. / jnp.sqrt(11.)
        self.pts += expand_sign([r, r, s]) + expand_sign([r, s, r]) + expand_sign([s, r, r])
        self.pts = jnp.array(self.pts)
        self.coefs = self.coefs[self.np]
        self.pts = self.pts[:self.np, :]


class Icosahedron(Quadrature):
    def __init__(self, n_p):
        super(Icosahedron, self).__init__(n_p)

        A_num = 2
        B_num = 10
        C_num = 20

        self.coefs = {
            12: jnp.array([1. / 12.] * (A_num + B_num)),
            32: jnp.array([5. / 168.] * (A_num + B_num) + [27. / 840.] * C_num)
        }

        polars = [[0, 0], [jnp.pi, 0]]
        polars += [[jnp.arctan(2), 2 * k * jnp.pi / 5] for k in range(5)]
        polars += [[jnp.pi - jnp.arctan(2), (2 * k + 1) / 5. * jnp.pi] for k in range(5)]
        down = jnp.sqrt(15 + 6 * jnp.sqrt(5))
        theta1 = jnp.arccos((2 + jnp.sqrt(5)) / down)
        theta2 = jnp.arccos(1. / down)

        polars += [[theta1, (2 * k + 1) * jnp.pi / 5.] for k in range(5)]
        polars += [[theta2, (2 * k + 1) * jnp.pi / 5.] for k in range(5)]
        polars += [[jnp.pi - theta1, 2 * k * jnp.pi / 5] for k in range(5)]
        polars += [[jnp.pi - theta2, 2 * k * jnp.pi / 5] for k in range(5)]

        toCartesian = lambda p: [jnp.sin(p[0]) * jnp.cos(p[1]), jnp.sin(p[0]) * jnp.sin(p[1]), jnp.cos(p[0])]

        self.pts = jnp.array([toCartesian(polar) for polar in polars])[:self.np, :]
        self.coefs = self.coefs[self.np]
