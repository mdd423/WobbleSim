import numpy as np
import numpy.random as random
import sys
import h5py

import scipy.ndimage as img
import scipy.interpolate as interp

import astropy.units as u
import astropy.constants as const
import astropy.table as at
import astropy.io
import scipy.io
import io

import data.tellurics.skycalc.skycalc as skycalc
import data.tellurics.skycalc.skycalc_cli as sky_cli
import json
import requests

def lanczos_interpolation(x,xs,ys,a=4):
    # x = x * u.radian
    y = np.zeros(x.shape)
    for i,x_value in enumerate(x):
        samples = np.arange(np.floor(x_value) - a + 1,np.floor(x_value) + a,dtype=int)
        for j,sample in enumerate(samples):
            y[i] += ys[j] * lanczos_kernel(x_value - j,a)
    return y

def lanczos_kernel(x,a):
    if x == 0:
        return 1
    if x > -a and x < a:
        return a*np.sinc(np.pi*u.radian*x) * np.sinc(np.pi*u.radian*x/a)/(np.pi**2 * x**2)
    return 0

def same_dist_elems(arr):
    diff = arr[1] - arr[0]
    for x in range(1, len(arr) - 1):
        if arr[x + 1] - arr[x] != diff:
            return False
    return True

def main(low_resolution,s2n,epoches,vp,epsilon,gamma,w):
    generate_data = False

    # Read In Files And Simulate
    #################################################
    # lambda is in Angstroms here
    lamb_min = 5000
    lamb_max = 6300
    xmin = np.log(lamb_min)
    xmax = np.log(lamb_max)
    flux_stellar, lamb_stellar, deltas = read_in_stellar('data/stellar/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits',
                                                         'data/stellar/PHOENIX/lte02400-0.50-4.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits',
                                                         lamb_min,lamb_max,epoches)

    # lambda here is in nanometers
    try:
        trans_tellurics, lamb_tellurics, airmass = simulate_tellurics(inputFilename='data/tellurics/skycalc/skycalc_defaults.txt',
                                                                        almFilename='data/tellurics/skycalc/almanac_example.txt',
                                                                        epoches=epoches)
    except requests.exceptions.ConnectionError:
        trans_tellurics, lamb_tellurics, airmass = read_in_tellurics('data/tellurics/skycalc/output.txt')
        epochs = 1
        print('skycalc needs network connection...\n using local file')
    # lambda is in nanometers here as well
    trans_gas, lamb_gas = read_in_gas_cell(filename='data/gascell/keck_fts_renorm.idl')

    lamb_tellurics *= u.nm
    lamb_gas       *= u.Angstrom
    lamb_stellar   *= u.Angstrom

    x_s = np.log(lamb_stellar/u.Angstrom)
    x_t = np.log(lamb_tellurics/u.Angstrom)
    x_g = np.log(lamb_gas/u.Angstrom)
    median_diff = min([get_median_difference(x) for x in [x_s,x_t[0],x_g]])

    # print(same_dist_elems(x_t[0,:]),same_dist_elems(x_s[:]),same_dist_elems(x_g))
    # sys.exit()

    xs = np.arange(np.log(lamb_min),np.log(lamb_max),step=median_diff)
    f_s   = np.empty((epoches,xs.shape[0]))
    f_t   = np.empty((epoches,xs.shape[0]))
    f_g   = interpolate(xs,x_g,trans_gas)

    f_sum = np.empty((epoches,xs.shape[0]))
    for i in range(epoches):
        f_s[i,:]   = interpolate(xs + deltas[i],x_s,     flux_stellar)
        f_t[i,:]   = interpolate(xs,x_t[i,:],trans_tellurics[i])
        f_sum[i,:] = f_s[i,:] * f_t[i,:] * f_g

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
    g_l = np.linspace(-2,2,5)
    g_l = gauss_func(g_l,mu=0.0,sigma=1.0)
    g_l = g_l/np.linalg.norm(g_l,ord=1)
    f_tot = np.apply_along_axis(img.convolve,0,f_sum,g_l) # convolve just tell star and gas

    # Generate dataset grid & jitter & stretch
    ##################################################
    x   = np.arange(xmin,xmax,step=spacing_from_res(low_resolution))
    nlr = x.shape[0]
    x_hat, m    = stretch(x,epoches,epsilon)
    x_hat, delt = jitter(x,epoches,w)

    # Interpolate Spline and Add Noise
    ##################################################
    s2n   = get_s2n(x_hat.shape,s2n)
    f_ds  = np.empty(x_hat.shape)
    noise = np.empty(x_hat.shape)
    for i in range(f_ds.shape[0]):
        f_ds[i,:] = interpolate(x_hat[i,:],xs,f_tot[i,:])
        for j in range(f_ds.shape[1]):
            f_ds[i,j] *= random.normal(1,1./s2n[i,j])

    # Get Error Bars
    ###################################################
    ferr_out = generate_errors(f_ds,s2n,gamma)
    lmb_out  = np.exp(x)

    # Pack Output into Dictionary
    ###################################################
    out = {"wavelength_sample":lmb_out,
            "flux":f_ds,
            "flux_error":ferr_out,
            "wavelength_theory":np.exp(xs),
            "flux_tellurics":f_t,
            "flux_stellar":f_s,
            "flux_gas":f_g,
            "del":delt,
            "m":m,
            "airmass":airmass,
            "delta":deltas}

    return out

def get_skycalc_defaults(inputFilename,almFilename,isVerbose=False):
    dic = {}

    # Query the Almanac if alm option is enabled
    if almFilename:

        # Read the input parameters
        inputalmdic = None
        try:
            with open(almFilename, 'r') as f:
                inputalmdic = json.load(f)
        except ValueError:
            with open(almFilename, 'r') as f:
                inputalmdic = sky_cli.loadTxt(f)

        if not inputalmdic:
            raise ValueError('Error: cannot read' + almFilename)

        alm = skycalc.AlmanacQuery(inputalmdic)
        dic = alm.query()

    if isVerbose:
        print('Data retrieved from the Almanac:')
        for key, value in dic.items():
            print('\t' + str(key) + ': ' + str(value))

    if inputFilename:

        # Read input parameters
        inputdic = None
        try:
            with open(inputFilename, 'r') as f:
                inputdic = json.load(f)
        except ValueError:
            with open(inputFilename, 'r') as f:
                inputdic = sky_cli.loadTxt(f)

        if not inputdic:
            raise ValueError('Error: cannot read ' + inputFilename)

        # Override input parameters
        if isVerbose:
            print('Data overridden by the user\'s input file:')
        for key, value in inputdic.items():
            if isVerbose and key in dic:
                print('\t' + str(key) + ': ' + str(value))
            dic[key] = value

    # Fix the observatory to fit the backend
    try:
        dic = sky_cli.fixObservatory(dic)
    except ValueError:
        raise

    if isVerbose:
        print('Data submitted to SkyCalc:')
        for key, value in dic.items():
            print('\t' + str(key) + ': ' + str(value))

    return dic

def sample_deltas(epoches,vel_width=30*u.km/u.s):
    deltas  = np.array(shifts((2*random.rand(epoches)-1)*vel_width))
    return deltas

def read_in_stellar(wavefile,fluxfile,lamb_min,lamb_max,epoches,vel_width=30*u.km/u.s):
    deltas = sample_deltas(epoches,vel_width)
    lambdas_all = astropy.io.fits.open(wavefile)['PRIMARY'].data
    flux_all    = astropy.io.fits.open(fluxfile)['PRIMARY'].data
    return flux_all, lambdas_all, deltas

def sample_airmass(epoches):
    airmass = (2 * random.rand(epoches)) + 1
    return airmass

def simulate_tellurics(inputFilename,almFilename,epoches):
    airmass = sample_airmass(epoches)
    dic     = get_skycalc_defaults(inputFilename,almFilename)
    lambda_grid       = []
    transmission_grid = []
    for a in airmass:
        dic['airmass'] = a
        skyModel = skycalc.SkyModel()
        skyModel.callwith(dic)

        data = skyModel.getdata()
        data = io.BytesIO(data)
        hdu = at.Table.read(data)
        transmission_grid.append(hdu['trans'].data)
        lambda_grid.append(hdu['lam'].data)
    return np.array(transmission_grid), np.array(lambda_grid), np.array(airmass)

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

def read_in_gas_cell(filename='data/gascell/keck_fts_renorm.idl'):
    array = scipy.io.readsav(filename)
    transmission = array['siod']
    wavelength   = array['wiod']
    return transmission, wavelength

def get_median_difference(x):

    return np.median([t - s for s, t in zip(x, x[1:])])

def interpolate(x,xs,ys):
    spline = interp.CubicSpline(xs,ys)
    return spline(x)

def get_s2n(shape,constant):
    return np.ones(shape) * constant

def zplusone(vel):
    return np.sqrt((1 + vel/(const.c))/(1 - vel/(const.c)))

def shifts(vel):
    return np.log(zplusone(vel))

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
