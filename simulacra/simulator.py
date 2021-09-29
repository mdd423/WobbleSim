import numpy as np
import numpy.random as random

import sys
import h5py

import scipy.ndimage as img
import scipy.interpolate as interp
import scipy.io

import astropy.units as u
import astropy.constants as const
import astropy.table as at
import astropy.io

import data.tellurics.skycalc.skycalc as skycalc
import data.tellurics.skycalc.skycalc_cli as sky_cli

import json
import requests
import io

    # WobbleSim/wobblesim.py
    # WobbleSim/cli/main.py


def main(low_resolution,s2n,epoches,vp,epsilon,gamma,w,stellarname_wave,stellarname_flux,skycalcname,skycalcalma,gascellname,a):
    generate_data = False

    # Read In Files And Simulate
    #################################################
    # lambda is in Angstroms here
    lamb_min = 5000
    lamb_max = 6300
    xmin   = np.log(lamb_min)
    xmax   = np.log(lamb_max)
    deltas = sample_deltas(epoches,vel_width=vp*u.km/u.s)
    flux_stellar, lamb_stellar = read_in_stellar(stellarname_wave,
                                                 stellarname_flux)

    # lambda here is in nanometers
    trans_tellurics, lamb_tellurics, airmass = simulate_tellurics(inputFilename=skycalcname,
                                                                    almFilename=skycalcalma,
                                                                    epoches=epoches)

    # lambda is in nanometers here as well
    trans_gas, lamb_gas = read_in_gas_cell(filename=gascellname)

    lamb_tellurics *= u.nm
    lamb_gas       *= u.Angstrom
    lamb_stellar   *= u.Angstrom

    x_s = np.log(lamb_stellar/u.Angstrom)
    x_t = np.log(lamb_tellurics/u.Angstrom)
    x_g = np.log(lamb_gas/u.Angstrom)

    temp = [get_median_difference(x) for x in [x_s,x_t[0],x_g]]
    median_diff = min(temp)

    xs  = np.arange(np.log(lamb_min),np.log(lamb_max),step=median_diff)
    f_s = np.empty((epoches,xs.shape[0]))
    f_t = np.empty((epoches,xs.shape[0]))
    f_g = interpolate(xs,x_g,trans_gas)

    f_theory = np.empty((epoches,xs.shape[0]))
    for i in range(epoches):
        f_s[i,:]   = interpolate(xs + deltas[i],x_s,     flux_stellar)
        f_t[i,:]   = interpolate(xs,x_t[i,:],trans_tellurics[i])
        f_theory[i,:] = f_s[i,:] * f_t[i,:] * f_g

    # now take lambda grids from all of these and make a new one with
    # spacing equal to the minimum median spacing of the above grids
    # then using lanczos 5 interpolation interpolate all values onto new grids
    # then multiply element wise for combined theoretical spectrum

    if generate_data:
        # Initialize constants
        #################################################
        high_spacing = spacing_from_res(high_resolution)
        xs           = np.arange(xmin,xmax,step=high_spacing)
        line_width   = spacing_from_res(low_resolution)

        # Generate Theoretical Y values
        #################################################
        y_star,deltas  = generate_stellar(  sn,epoches,xs,line_width,ymin,ymax,vp*u.km/u.s)
        y_tell,airmass = generate_tellurics(tn,epoches,xs,line_width,ymin,ymax)
        y_gas ,mu_g    = generate_gas_cell( gn,epoches,xs,line_width,ymin,ymax)

        y_sum = y_star + y_tell + y_gas
        f_sum = np.exp(y_sum)

    # Convolve with Telescope PSF
    ################################################
    lsf = mean_lsf(low_resolution,median_diff,sigma_range=5.0)
    f_lsf = np.empty(f_theory.shape)
    for iii in range(f_theory.shape[0]):
        f_lsf[iii,:] = img.convolve(f_theory[iii,:],lsf)
        # np.apply_along_axis(dummy,1,f_theory,lsf) # convolve just tell star and gas

    # Generate dataset grid & jitter & stretch
    ##################################################
    x   = np.arange(xmin,xmax,step=spacing_from_res(low_resolution))
    nlr = x.shape[0]
    x_hat, m    = stretch(x,epoches,epsilon)
    x_hat, delt = jitter(x,epoches,w)

    # Interpolate Spline and Add Noise
    ##################################################
    s2n_grid  = get_s2n(x_hat.shape,s2n)
    f_exp     = np.empty(x_hat.shape)
    f_readout = np.empty(x_hat.shape)
    # noise = np.empty(x_hat.shape)
    for i in range(f_exp.shape[0]):
        f_exp[i,:] = lanczos_interpolation(x_hat[i,:],xs,f_lsf[i,:],dx=median_diff,a=a)
        for j in range(f_exp.shape[1]):
            f_readout[i,j] = f_exp[i,j] * random.normal(1,1./s2n_grid[i,j])

    # Get Error Bars
    ###################################################
    ferr_out = generate_errors(f_readout,s2n_grid,gamma)
    lmb_out  = np.exp(x)

    # Pack Output into Dictionary
    ###################################################
    out = {"wavelength_sample":lmb_out,
            "flux":f_readout,
            "flux_error":ferr_out,
            "wavelength_theory":np.exp(xs),
            "flux_tellurics":f_t,
            "flux_stellar":f_s,
            "flux_gas":f_g,
            "flux_lsf":f_lsf,
            "del":delt,
            "m":m,
            "airmass":airmass,
            "delta":deltas}

    return out

def mean_lsf(low_resolution,spacing,sigma_range=5.0):
    lsf = np.arange(-sigma_range/low_resolution,sigma_range/low_resolution,step=spacing)
    lsf = gauss_func(lsf,mu=0.0,sigma=1.0/low_resolution)
    lsf /= np.linalg.norm(lsf,ord=1)
    return lsf

def lanczos_interpolation(x,xs,ys,dx,a=4):
    x0 = xs[0]
    y = np.zeros(x.shape)
    for i,x_value in enumerate(x):
        # which is basically the same as sample=x[j-a+1] to x[j+a]
        # where j in this case is the nearest index xs_j to x_value
#         print("value: ", x_value)
#         print("closest: ",xs[int((x_value-x0)//dx)])
#         print,x_value)
        sample_min,sample_max = max(0,abs(x_value-x0)//dx - a + 1),min(xs.shape[0],abs(x_value-x0)//dx + a)

        samples = np.arange(sample_min,sample_max,dtype=int)
#         print(sample_min,sample_max)
        for sample in samples:
            y[i] += ys[sample] * lanczos_kernel((x_value - xs[sample])/dx,a)
    return y

def lanczos_kernel(x,a):
    if x == 0:
        return 1
    if x > -a and x < a:
        return a*np.sin(np.pi*u.radian*x) * np.sin(np.pi*u.radian*x/a)/(np.pi**2 * x**2)
    return 0

def same_dist_elems(arr):
    diff = arr[1] - arr[0]
    for x in range(1, len(arr) - 1):
        if arr[x + 1] - arr[x] != diff:
            return False
    return True

def read_in_tellurics(filename):
    hdu = astropy.io.fits.open(filename)
    print(hdu)
    prim = hdu['PRIMARY'].header.keys
    print(prim)
    sys.exit()
    prim = at.Table.read(hdu['PRIMARY'])
    print(prim.info())
    tbl  = at.Table.read(hdu[1])

    sys.exit()
    trans_grid = np.array(tbl['trans'].data)
    lamb_grid  = np.array(tbl['lam'].data)
    airmass = np.array(prim['airmass'].data)

    trans_grid = np.expand_dims(trans_grid,0)
    lamb_grid  = np.expand_dims(lamb_grid,0)
    return trans_grid, lamb_grid, airmass

def get_s2n(shape,constant):
    return np.ones(shape) * constant

def average_difference(x):
    return np.mean([t - s for s, t in zip(x, x[1:])])

def jitter(x,epoches,w=1.0):
    out = x
    if len(out.shape) == 1:
        out = np.expand_dims(out,axis=0)
        out = np.repeat(out,repeats=epoches,axis=0)

    width = average_difference(out[0,:])
    jitter = (2*random.rand(epoches) - 1) * width * w
    for i,delt in enumerate(jitter):
        out[i,:] += delt
    return out,jitter

def stretch(x,epoches,epsilon=0.01):
    if len(x.shape) == 1:
        x = np.expand_dims(x,axis=0)
        x = np.repeat(x,repeats=epoches,axis=0)
    m = (epsilon * (2*random.rand(epoches) - 1)) + 1
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
    return y, mus

def generate_errors(f,s2n,gamma=1.0):
    xs,ys = np.where(f < 0)
    for x,y in zip(xs,ys):
        f[x,y] = 0
    f_err = np.empty(f.shape)
    for i in range(f_err.shape[0]):
        for j in range(f_err.shape[1]):
            error = random.normal(scale=f[i,j]/s2n[i,j] * gamma)
            # print(error)
            f_err[i,j] = error
    return f_err
