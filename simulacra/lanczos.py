import numpy as np
import astropy.units as u
from functools import partial
import jax
import jax.numpy as jnp

# def lanczos_interpolation(x,xs,ys,dx,a=4):
#     x0 = xs[0]
#     y = np.zeros(x.shape)
#     # v_lanczos = np.vectorize(lanczos_kernel)
#     for i,x_value in enumerate(x):
#         # which is basically the same as sample=x[j-a+1] to x[j+a]
#         # where j in this case is the nearest index xs_j to x_value
#         sample_min,sample_max = max(0,abs(x_value-x0)//dx - a + 1),min(xs.shape[0],abs(x_value-x0)//dx + a)
#         samples = np.arange(sample_min,sample_max,dtype=int)
#         # y[i]/ = v_lanczos((x_value - xs[samples])/dx,a).sum()
#         for sample in samples:
#             y[i] += ys[sample] * lanczos_kernel((x_value - xs[sample])/dx,a)
#     return y

@partial(jnp.vectorize,excluded=(1,))
def lanczos_kernel(x,a):
    return jnp.where(x == 0, 1, \
                     jnp.where((x > -a) & (x < a), \
                               a * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * x / a) / (jnp.pi**2 * x**2), 0.0))

def lanczos_matrix(x,xs,dx,a=4):
    return jnp.where(((x[None,:] - xs[:,None])/dx < a)*((x[None,:] - xs[:,None])/dx > -a),\
                     lanczos_kernel((x[None,:] - xs[:,None])/dx,a),0.0)

def lanczos_interpolation(x,xs,ys,dx,a=4):
    M = lanczos_matrix(xs,xs,dx,a)
    print("created matrix")
    theta = jnp.linalg.solve(M, ys)
    print("solved matrix")
    return theta @ lanczos_matrix(xs, x, dx, a)

