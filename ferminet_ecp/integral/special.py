# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import jax.numpy as jnp


def legendre(x, l):
    if l == 0:
        return jnp.ones_like(x)
    if l == 1:
        return x
    if l == 2:
        return (3 * x**2 - 1) / 2
    else:
        pass
