import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as img

import astropy.units as u
import astropy.constants as const
import numpy.random as random

import scipy.interpolate as interp

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--hr',action='store',default=100000,type=np.float32)
parser.add_argument('--lr',action='store',default=20000,type=np.float32)
parser.add_argument('--gn',action='store',default=10,type=np.int)
parser.add_argument('--sn',action='store',default=10,type=int)
parser.add_argument('--tn',action='store',default=10,type=int)
parser.add_argument('--xmin',action='store',default=9.53,type=np.float32)
parser.add_argument('--xmax',action='store',default=9.82,type=np.float32)
parser.add_argument('--ymin',action='store',default=0.0,type=np.float32)
parser.add_argument('--ymax',action='store',default=0.9,type=np.float32)
parser.add_argument('--vp',action='store',default=30*u.km/u.s,type=np.float32)
parser.add_argument('--w',action='store',default=1.0,type=np.float32)
parser.add_argument('--epsilon',action='store',default=0.0001,type=np.float32)
parser.add_argument('--gamma',action='store',default=0.001,type=np.float32)
parser.add_argument('--s2n',action='store',default=100,type=np.float32)
parser.add_argument('--epoches',action='store',default=30,type=int)
args   = parser.parse_args()

def main():
    high_resolution = args.hr
    high_spacing = spacing_from_res(high_resolution)
    xmin = args.xmin
    xmax = args.xmax
    xs = np.arange(xmin,xmax,step=high_spacing)
    s2n = args.s2n
    epoches = args.epoches
    low_resolution = args.hr
    line_width = spacing_from_res(low_resolution)

    # Generate Theoretical Y values
    #################################################
    y_star,deltas  = generate_stellar(args.sn,epoches,xs,line_width,args.ymin,args.ymax,args.vp)
    y_tell,airmass = generate_tellurics(args.tn,epoches,xs,line_width,args.ymin,args.ymax)
    y_gas          = generate_gas_cell(args.gn,epoches,xs,line_width,args.ymin,args.ymax)

    y_sum = y_star + y_tell + y_gas
    f_sum = np.exp(y_sum)

    # Convolve with Telescope PSF
    ################################################

    g_l = np.linspace(-2,2,5)
    g_l = gauss_func(g_l,mu=0.0,sigma=1.0)
    g_l = [0.25,0.5,0.25]
    f_tot = np.apply_along_axis(img.convolve,0,f_sum,g_l) # convolve just tell star and gas

    # Interpolate Spline
    ##################################################
    spline = []
    for i in range(epoches):
        spline.append(interp.CubicSpline(xs,f_tot[i,:]))

    # Generate dataset grid & jitter & stretch
    ##################################################
    x       = np.arange(xmin,xmax,step=spacing_from_res(low_resolution))
    nlr = x.shape[0]
    x_hat, delt = jitter(x,epoches,args.w)
    x_hat, m    = stretch(x,epoches,args.epsilon)
    # x_hat = x

    f_ds = np.empty((epoches,nlr))
    for i in range(epoches):
        f_ds[i,:]= spline[i](x_hat[i,:])

    # Add noise
    ###################################################
    noise = random.normal(1,1./s2n,size=f_ds.shape)
    f_out = f_ds * noise
    ferr_out = generate_errors(f_ds,args.gamma)
    lmb_out = np.exp(x)

    # Plot an epoch
    ####################################################
    epoch_idx = 17
    plt.figure(figsize=(20,8))
    plt.title('wobble toy data')
    plt.xlabel('$\lambda_{%i}$' % epoch_idx)
    plt.ylabel('$f_{%i}$' % epoch_idx)
    plt.plot(np.exp(xs),np.exp(y_star[epoch_idx,:]),'red',alpha=0.5,label='star')
    plt.plot(np.exp(xs),np.exp(y_tell[epoch_idx,:]),'blue',alpha=0.5,label='telluric')
    plt.plot(np.exp(xs),np.exp(y_gas[epoch_idx,:]),'green',alpha=0.5,label='gas cell')
    plt.errorbar(lmb_out,f_out[epoch_idx,:],ferr_out[epoch_idx,:],fmt='.k',elinewidth=0.7,zorder=1,alpha=0.5,ms=6,label='data')
    plt.legend()
    plt.show()

def zplusone(vel):
    return np.sqrt((1 + vel/(const.c))/(1 - vel/(const.c)))

def shifts(vel):
    return np.log(zplusone(vel))

def average_difference(x):
    return np.mean([t - s for s, t in zip(x, x[1:])])

def jitter(x,epoches,w=1.0):
    if len(x.shape) == 1:
        x = np.expand_dims(x,axis=0)
        x = np.repeat(x,repeats=epoches,axis=0)
    width = average_difference(x[0,:])
    jitter = (2*random.rand(epoches) - 1) * width * w
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
