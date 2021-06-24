import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as img

# import dataset as wobble_data
import astropy.units as u
import astropy.constants as const
import numpy.random as random

import scipy.interpolate as interp

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-hr',action='store',default=100000,type=float32)
parser.add_argument('-lr',action='store',default=20000,type=float32)
parser.add_argument('-gn',action='store',default=10,type=int)
parser.add_argument('-sn',action='store',default=10,type=int)
parser.add_argument('-tn',action='store',default=10,type=int)
args   = parser.parse_args()


def main():
    high_resolution = args.hr
    high_spacing = spacing_from_res(high_resolution)
    xmin = 9.53
    xmax = 9.82
    xs = np.arange(xmin,xmax,step=high_spacing)
    s2n = 100
    epoches = 30
    n_lines = 10
    low_resolution = args.hr
    line_width = spacing_from_res(low_resolution)
    # print(line_width.unit)

    y_star,deltas  = generate_stellar(n_lines,epoches,xs,line_width)
    y_tell,airmass = generate_tellurics(n_lines,epoches,xs,line_width)
    y_gas          = generate_gas_cell(n_lines,epoches,xs,line_width)

    y_sum = y_star + y_tell + y_gas
    f_sum = np.exp(y_sum)

    g_l = np.linspace(-2,2,5)
    g_l = gauss_func(g_l,mu=0.0,sigma=1.0)
    g_l = [0.25,0.5,0.25]
    f_tot = np.apply_along_axis(img.convolve,0,f_sum,g_l) # convolve just tell star and gas

    spline = []
    for i in range(epoches):
        spline.append(interp.CubicSpline(xs,f_tot[i,:]))

    x = np.arange(xmin,xmax,step=spacing_from_res(low_resolution))

    x, delt = jitter(x,epoches)
    x, m  = stretch(x,epoches)

    f_ds = np.empty((epoches,x.shape[1]))
    print(len(spline))
    for i in range(epoches):
        f_ds[i,:]= spline[i](x[i,:])
    # f_err = generate_errors(f_tot,0.001)

    noise = random.normal(1,1./s2n,size=f_ds.shape)
    f_out = f_ds * noise

    lmb_out = np.exp(x)

    epoch_idx = 17
    ferr_out = generate_errors(f_ds,0.001)
    plt.figure(figsize=(20,8))
    plt.errorbar(lmb_out[epoch_idx,:],f_out[epoch_idx,:],ferr_out[epoch_idx,:],fmt='.k',elinewidth=0.7,zorder=1,alpha=0.5,ms=6)
    plt.show()

def zplusone(vel):
    return np.sqrt((1 + vel/(const.c))/(1 - vel/(const.c)))

def shifts(vel):
    return np.log(zplusone(vel))

def average_difference(x):
    return np.mean([t - s for s, t in zip(x, x[1:])])

def jitter(x,epoches,tuning=1.0):
    if len(x.shape) == 1:
        x = np.expand_dims(x,axis=0)
        x = np.repeat(x,repeats=epoches,axis=0)
    width = average_difference(x[0,:])
    jitter = (2*random.rand(epoches) - 1) * width * tuning
    for i,delt in enumerate(jitter):
        x[i,:] += delt
    return x,jitter

def stretch(x,epoches,epsilon=0.01):
    if len(x.shape) == 1:
        x = np.expand_dims(x,axis=0)
        x = np.repeat(x,repeats=epoches,axis=0)
    m = (epsilon * (random.rand(epoches) - 0.5)) + 1
    for i,ms in enumerate(m):
        x[i,:] *= ms
    return x,m

def gauss_func(x,mu,sigma):
    return np.exp((-1/2)*(x - mu)**2/(sigma**2))

def generate_noise(epoches,size,scale=0.01):
    return random.normal(scale=scale,size=(epoches,size))

def spacing_from_res(R):
    return np.log(1+1/R)

def generate_stellar(n_lines,epoches,x,line_width,y_min=0.0,y_max=0.7,vel_width=30*u.km/u.s):
    deltas  = np.array(shifts((2*random.rand(epoches)-1)*vel_width))
    mus     = (np.max(x) - np.min(x))*random.rand(n_lines) + np.min(x)
    heights = (y_max - y_min) * random.rand(n_lines) + y_min

    y = np.zeros((epoches,x.shape[0]))
    for i,delta in enumerate(deltas):
        for j in range(n_lines):
            y[i,:] -= heights[j] * gauss_func(x + delta,mus[j],line_width)
    return y, deltas

def generate_tellurics(n_lines,epoches,x,line_width,y_min=0.0,y_max=0.7):
    airmass = random.rand(epoches)
    mus     = (np.max(x) - np.min(x))*random.rand(n_lines) + np.min(x)
    heights = (y_max - y_min) * random.rand(n_lines) + y_min

    y = np.zeros((epoches,x.shape[0]))
    for i in range(epoches):
        for j in range(n_lines):
            y[i,:] -= airmass[i] * heights[j] * gauss_func(x,mus[j],line_width)
    return y, airmass

def generate_gas_cell(n_lines,epoches,x,line_width,y_min=0.0,y_max=0.7):
    mus     = (np.max(x) - np.min(x))*random.rand(n_lines) + np.min(x)
    heights = (y_max - y_min) * random.rand(n_lines) + y_min

    y = np.zeros((epoches,x.shape[0]))
    for i in range(epoches):
        for j in range(n_lines):
            y[i,:] -= heights[j] * gauss_func(x,mus[j],line_width)
    return y

def generate_errors(f,w=0.1):
    f_err = np.empty(f.shape)
    for i in range(f_err.shape[0]):
        for j in range(f_err.shape[1]):
            f_err[i,j] = random.normal(scale=np.sqrt(w*f[i,j]))
    return f_err

if __name__ == '__main__':
    main()
