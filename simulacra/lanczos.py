import numpy as np
import astropy.units as u

def lanczos_interpolation(x,xs,ys,dx,a=4):
    x0 = xs[0]
    y = np.zeros(x.shape)
    v_lanczos = np.vectorize(lanczos_kernel)
    for i,x_value in enumerate(x):
        # which is basically the same as sample=x[j-a+1] to x[j+a]
        # where j in this case is the nearest index xs_j to x_value
        sample_min,sample_max = max(0,abs(x_value-x0)//dx - a + 1),min(xs.shape[0],abs(x_value-x0)//dx + a)
        samples = np.arange(sample_min,sample_max,dtype=int)
        # y[i]/ = v_lanczos((x_value - xs[samples])/dx,a).sum()
        for sample in samples:
            y[i] += ys[sample] * lanczos_kernel((x_value - xs[sample])/dx,a)
    return y

def lanczos_kernel(x,a):
    if x == 0:
        return 1
    if x > -a and x < a:
        return a*np.sin(np.pi*u.radian*x) * np.sin(np.pi*u.radian*x/a)/(np.pi**2 * x**2)
    return 0
