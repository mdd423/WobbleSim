import numpy as np

def hermitegaussian(coeffs,x,sigma):
    xhat = (x/sigma)
    herms = np.polynomial.hermite.Hermite(coeffs)
    return herms(xhat) * np.exp(-xhat**2)

def continuous_convolve(kernels,obj):
    out = np.empty(obj.shape)
    for i in range(kernels.shape[0]):
        out[jj] = np.dot(obj[max(0,jj-centering):min(size,jj+n-centering)], kernels[i,max(0,centering-jj):min(n,size-jj+centering)])
    return out

def convolve_hermites(f_in,coeffs,center_kw,sigma,sigma_range,spacing):
    x = np.arange(-sigma_range * max(sigma),sigma_range * max(sigma),step=spacing)

    if center_kw == 'centered':
        centering = int(x.shape[0]/2)
    elif center_kw == 'right':
        centering = 0
    elif center_kw == 'left':
        centering = x.shape[0]-1
    else:
        print('setting lsf centering to middle')
        centering = int(x.shape[0]/2)

    f_out = np.empty(f_in.shape)
    size = f_in.shape[0]
    n    = x.shape[0]

    for jj in range(f_out.shape[0]):
        kernel = hermitegaussian(coeffs[jj,:],x,sigma[jj])
        # L1 normalize the kernel so the total flux is conserved
        kernel /= np.sum(kernel)
        f_out[jj] = np.dot(f_in[max(0,jj-centering):min(size,jj+n-centering)]\
                                    ,kernel[max(0,centering-jj):min(n,size-jj+centering)])
    return f_out
