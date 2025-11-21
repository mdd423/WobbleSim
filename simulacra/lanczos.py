import numpy as np
import astropy.units as u
from functools import partial
import jax
import jax.numpy as jnp


@partial(jnp.vectorize, excluded=(1,))
def lanczos_kernel(x, a):
    '''Lanczos kernel function.
    Parameters
    ----------
    x : float
        The input value.
    a : int
        The Lanczos parameter.
    Returns
    -------
    float
        The value of the Lanczos kernel at x.
    '''
    return jnp.where(
        x == 0,
        1,
        jnp.where(
            (x >= -a) & (x <= a),
            a * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * x / a) / (jnp.pi**2 * x**2),
            0.0,
        ),
    )


def lanczos_matrix(x, xs, dx, a=4):
    '''Constructs the Lanczos interpolation matrix.
    Parameters 
    ----------
    x : array-like
        The input values where the kernel is evaluated.
    xs : array-like
        The sample points.
    dx : float
        The spacing between sample points.
    a : int
        The Lanczos parameter.
    Returns
    -------
    array-like
        The Lanczos interpolation matrix.
    '''
    return jnp.where(
        ((x[None, :] - xs[:, None]) / dx < a) * ((x[None, :] - xs[:, None]) / dx > -a),
        lanczos_kernel((x[None, :] - xs[:, None]) / dx, a),
        0.0,
    )


def lanczos_interpolation(x, xs, ys, dx, a=4):
    '''Performs Lanczos interpolation.
    Parameters
    ----------
    x : array-like
        The input values where interpolation is desired.
    xs : array-like
        The sample points.
    ys : array-like
        The sample values at the sample points.
    dx : float
        The spacing between sample points.
    a : int
        The Lanczos parameter.
    Returns
    -------
    array-like
        The interpolated values at x.
    '''
    print("creating lanczos matrix")
    M = lanczos_matrix(xs, xs, dx, a)
    print("solving lanczos matrix")
    theta = jnp.linalg.solve(M, ys)
    print("interpolating lanczos")
    return theta @ lanczos_matrix(x, xs, dx, a)
