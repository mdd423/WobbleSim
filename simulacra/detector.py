import numpy as np
import astropy.units as u
import astropy.constants as const
import astropy.time as at
import astropy.coordinates as coord

import sys

import scipy.interpolate as interp
import scipy.ndimage as img
import scipy.sparse
import jax.numpy as jnp
import jax.random
from functools import partial

import logging

from simulacra.dataset import DetectorData
import simulacra.lanczos
import simulacra.convolve
import simulacra.star


from itertools import repeat
from multiprocessing import Pool

PRNG_KEY = jax.random.PRNGKey(1010101)

def dict_of_attr(data,obj):
    obj_list = [a for a in dir(obj) if not a.startswith('__')]
    for ele in obj_list:
        try:
            data[ele] = getattr(obj, ele)
        except AttributeError:
            pass
    return data

def get_median_difference(x):

    return np.median([t - s for s, t in zip(x, x[1:])])


def average_difference(x):
    return np.mean([t - s for s, t in zip(x, x[1:])])

# def generate_errors(f,snr):
#     xs,ys = np.where(f < 0)
#     for x,y in zip(xs,ys):
#         f[x,y] = 0
#     f_err = np.empty(f.shape)
#     for i in range(f_err.shape[0]):
#         for j in range(f_err.shape[1]):
#             f_err[i,j] = f[i,j]/snr[i,j]
#     return f_err

# from numba import vectorize, float64

# @jnp.vectorize
# def generate_errors_v(f, snr):
#     return f / snr

# def add_noise(f_exp,snr_grid):
#     f_readout = np.empty(f_exp.shape)
#     for i in range(f_exp.shape[0]):
#         print('snr {}: {}'.format(i,np.median(snr_grid[i,:])))
#         for j in range(f_exp.shape[1]):
#             f_readout[i,j] = f_exp[i,j] + random.normal(0.0,f_exp[i,j]/snr_grid[i,j])
#     return f_readout

def interpolate_mask(xs,mask_the,x_hat):
    return np.array([interp.interp1d(xs,mask_the[i,:].astype(float),kind='nearest')(x_hat[i,:]) for i in range(x_hat.shape[0])]).astype(bool)

def check_type(value,type):
    if isinstance(value,u.Quantity):
        if value.unit.physical_type in u.get_physical_type(type.unit):
            pass
        else:
            logging.error('must be in a units of {}.\nor will be assumed to be in units of photons per second.'.format(type))
    else:
        value *= type
    return value

def check_shape(value,shape):
    if hasattr(value,'shape'):
        if len(value.shape) == 0:
            return value * np.ones(shape)
        elif value.shape == shape:
            return value
    else:
        return value * np.ones(shape)


class Detector:
    def __init__(self,stellar_model,resolution,loc,area,wave_grid,through_put=0.2,wave_padding=5*u.Angstrom,a=4,*args,**kwargs):
        '''Detector model that simulates spectra from star given resolution...

        Detector takes on a given theoretical `stellar_model`, `resolution`, with
        a constant signal to noise ratio, `snr`. Then call simulate to generate a given number
        of epoches of spectral data.

        Parameters
        ----------
        stellar_model: TheoryModel with a generate_data method that returns flux, wave, and deltas
        resolution: the resolution of the detector being simulated
        epsilon: a small that is used to pull the random stretchs for the wavelength grid, m ~ uniform(1-epsilon,1+epsilon)
        snr: the constant signal to noise ratio
        gamma: a constant that random effects the errorbars of flux \sigma_f ~ normal(mu=0.0,sigma=gamma * f_i,j/snr_i,j)
        w: a constant that random affects the jitter to the wavelength grid, del ~ uniform(-w*pxl_width, w*pxl_width)
        a: the number of kernel functions used at the lanczos interpolation step

        Returns
        -------
        data: DetectorData type that contains all parameters generate
        '''
        # Simulator Models
        self.stellar_model = stellar_model
        self.transmission_models = []

        # Lanczos Parameters
        self.a       = a

        # Simulation Parameters
        self._lambmin = 0.0 * u.nm
        self._lambmax = 100000 * u.nm

        # Detector Properties
        self.wave_grid    = wave_grid
        self.wave_padding = wave_padding
        self.through_put  = through_put
        self.area = area
        self.loc  = loc
        # LSF properties
        self.sigma_range   = 5.0
        self.resolution   = resolution
        self.lsf_const_coeffs = [1.0]
        # self.sigma      = 1.0/resolution

        self.lsf_centering = 'centered'

        self.transmission_cutoff = 10.

    def res(self,wavelength):
        if isinstance(self._resolution, float):
            return self._resolution 
        elif hasattr(self._resolution, '__call__'):
            return self._resolution(wavelength)
        else:
            logging.error('resolution has not been set.')
            return 0

    def resolution():
        doc = "The resolution property."
        def fset(self, value):
            if isinstance(value,float):
                self._resolution = value
            elif hasattr(value, '__call__'):
                self._resolution = value
            else:
                logging.error('resolution grid needs to be a single value \
                                or a callable that takes wavelength as input')
        def fget(self):
            return self._resolution

        def fdel(self):
            logging.warn('overwriting resolution')
            del self._resolution
        return locals()
    resolution = property(**resolution())

    def lambmin():
        doc = "The lambmin property."
        def fget(self):
            return np.min(self.wave_grid) - self.wave_padding
        return locals()
    lambmin = property(**lambmin())

    def lambmax():
        doc = "The lambmax property."
        def fget(self):
            return np.max(self.wave_grid) + self.wave_padding
        return locals()
    lambmax = property(**lambmax())

    def wave_grid():
        doc = "The wave_grid property."
        def fget(self):
            return self._wave_grid
        def fset(self, new_grid):
            minimum = self.checkmin()
            maximum = self.checkmax()
            if minimum >= maximum:
                logging.error('no overlap between selected wave grids\nmodel cannot be added')
                self.transmission_models.pop()
                return
            self._wave_grid = new_grid[np.multiply(new_grid <= maximum, new_grid >= minimum,dtype=bool)]
            if np.min(new_grid) < minimum:
                print("wave_grid min -> {}".format(minimum))
            if np.max(new_grid) > maximum:
                print("wave_grid max -> {}".format(maximum))
        return locals()
    wave_grid = property(**wave_grid())

    def wave_difference():
        doc = "the wave_difference property"
        def fget(self):
            try:
                return self._wave_difference
            except AttributeError:
                diff = self.wave_grid[1:] - self.wave_grid[:-1]
                return np.concatenate(([np.mean(diff)],diff))
        def fset(self,value):
            if isinstance(value, np.ndarray):
                if value.shape == self.wave_grid.shape:
                    self._wave_difference = value
                else:
                    logging.error('difference array must have the same shape as wave grid {}.'.format(value.shape,self.wave_grid.shape))
            else:
                self._wave_difference = value * np.ones(self.wave_grid.shape)
        def fdel(self):
            self._wave_difference = None
        return locals()
    wave_difference = property(**wave_difference())

    def add_model(self,model):
        self.transmission_models.append(model)
        # after adding model to list
        # reset the wave_grid so that it can be truncated
        self.wave_grid = self.wave_grid

    def checkmin(self):
        minimums = [self.stellar_model.lambmin]
        for model in self.transmission_models:
            minimums += [model.lambmin]
        # print(minimums)
        return max(minimums)

    def checkmax(self):
        maximums = [self.stellar_model.lambmax]
        for model in self.transmission_models:
            maximums += [model.lambmax]
        # print(maximums)
        return min(maximums)

    def add_noise(self, f, snr):
        '''
            Add noise to the flux based on the signal to noise ratio. Vectorized by JAX.
            Parameters:
            f (np.ndarray) [float] flux array
            snr (np.ndarray) [float] signal to noise ratio
        '''

        return f + jax.random.normal(PRNG_KEY,shape=f.shape,dtype=f.dtype)*f/snr
    
    def generate_errors(self,f,snr):
        '''
            Generate errors based on the flux and signal to noise ratio.
            Parameters:
            f (np.ndarray) [float] flux array
            snr (np.ndarray) [float] signal to noise ratio
        '''
        
        return f / snr

    def simulate(self,obs_times,t_exp=None,snrs=None,wavelength_trigger=None,*args,**kwargs):
        '''
            The working function of the detector that creates the simulated data with the given
            parameters previously set.
            Parameters:
            obs_times (np.ndarray) [astropy.time.Time] midtime of exposure,
                used to determine star velocity for redshift
            EITHER
            t_exp (np.ndarray) [astropy.time.Time] length of time of exposure,
                to determine the number of photons, and signal to noise ratios
            OR
            {
            snrs (np.ndarray) [float] target snr of each epoch, the root of signal to noise function
                is found with respect to time (self.trigger)
            AND
            wavelength_trigger: either a single wavelength or wavelength range to take
                average over when finding length of exposure time
            }
        '''

        if len(self.wave_grid) == 0:
            print('wave_grid is empty')
            sys.exit()
        data = DetectorData()
        data['data'] = {}
        data['data']['obs_times'] = obs_times

        data['data']['epoches']   = obs_times.shape[0]
        epoches = obs_times.shape[0]

        if snrs is not None:
            if wavelength_trigger is None:
                wavelength_trigger = self.wave_grid
                # wavelength_trigger = (self.wave_grid[0] + self.wave_grid[-1])/2

            if not hasattr(snrs,'__iter__'):
                snrs = snrs * np.ones(epoches)
            data['data']['snrs'] = snrs
            data['data']['wavelength_trigger'] = wavelength_trigger

        # Generate Stellar Spectra
        ###################################################
        data['theory'] = {}
        data['theory']['star'] = {}
        rvs = self.stellar_model.get_velocity(self,at.Time(obs_times))
        deltas = simulacra.star.shifts(rvs)
        flux_stellar, wave_stellar = self.stellar_model.get_spectra(self,obs_times)

        data['data']['rvs'], data['theory']['star']['deltas'] = rvs, deltas
        # data['theory']['star']['flux'], data['theory']['star']['wave'] = flux_stellar, wave_stellar
        differences = [get_median_difference(np.log(wave_stellar.to(u.Angstrom).value))]

        # Generate Transmission
        ###################################################
        print('generating spectra...')
        trans_flux, trans_wave = [], []
        for model in self.transmission_models:
            # data['theory']['interpolated'][model._name] = {}
            
            flux, wave = model.generate_transmission(self.stellar_model,self,obs_times)
            trans_flux.append(flux), trans_wave.append(wave)
            differences += [get_median_difference(np.log(wave[iii][:].to(u.Angstrom).value)) for iii in range(len(wave))]
            # data['theory'][model._name]['flux'],data['theory'][model._name]['wave'] = flux, wave
            print(model, differences)
        new_step_size = min(differences)

        # Interpolate all models and combine onto detector
        # PARALLELIZE
        ##################################################################
        # data['theory']['interpolated']['star'] = {}
        xs = np.arange(np.log(self.lambmin.to(u.Angstrom).value),np.log(self.lambmax.to(u.Angstrom).value),step=new_step_size)
        print('interpolating spline...')
        stellar_arr = np.empty((epoches,xs.shape[0]))
        trans_arrs  = np.empty((len(self.transmission_models),epoches,xs.shape[0]))

        stellar_arr = self.interpolate_grid(np.add.outer(deltas, xs),np.outer(np.ones(epoches),np.log(wave_stellar.to(u.Angstrom).value)),flux_stellar.to(u.erg/u.s/u.cm**3).value)
        for j, model in enumerate(self.transmission_models):
            trans_arrs[j,:,:] = self.interpolate_grid(np.outer(np.ones(epoches),xs),[np.log(x.to(u.Angstrom).value) for x in trans_wave[j][:]],trans_flux[j])

        print('combining grids...')
        data['theory']['star']['flux'] = stellar_arr
        fs        = stellar_arr.copy()
        flux_unit = flux_stellar.unit
        mask_the  = np.zeros(fs.shape,dtype=bool)
        data['theory']['total'] = {}
        for j,model in enumerate(self.transmission_models):
            data['theory'][model._name] = {}
            # print("fs: ", fs.shape)
            fs *= trans_arrs[j,:,:]
            mask_the = (trans_arrs[j,:,:] > self.transmission_cutoff) | mask_the
            data['theory'][model._name]['flux'] = trans_arrs[j,:,:]
        data['theory']['total'] = {}
        data['theory']['total']['flux'] = fs
        data['theory']['total']['wave'] = np.exp(xs) * u.Angstrom
        data['theory']['total']['mask'] = mask_the

        # Convolving using Hermite Coeffs
        #################################################
        # should be an array that can vary over pixel j or hermite m
        print('convolving...')
        f_lsf = self.convolve(xs,fs,self.res)

        data['theory']['lsf'] = {}
        data['theory']['lsf']['flux'] = f_lsf

        # Generate transform wavelength grid using jitter & stretch
        ##################################################
        x                    = np.log(self.wave_grid.to(u.Angstrom).value)#np.arange(self.xmin,self.xmax,step=res_step_size)
        x_hat, wt_parameters = self.wave_transform(x,epoches)

        data['parameters'] = {}
        data['parameters']['wavetransform'] = wt_parameters

        print('xs: {} {}\nxhat: {} {}'.format(np.exp(np.min(xs)),np.exp(np.max(xs)),np.exp(np.min(x_hat)),np.exp(np.max(x_hat))))
        data_mask = interpolate_mask(xs,mask_the,x_hat)
        data['data']['mask'] = data_mask

        # Interpolate using Lanczos and Add Noise
        ##################################################
        print('interpolating lanczos...')
        f_exp = self.interpolate_data(x_hat,xs,f_lsf,new_step_size)


        # print('area: {}\t avg d lambda: {}\t avg lambda: {}\t avg exp times: {}'.format(self.area,np.mean(self.wave_difference),np.mean(self.wave_grid),np.mean(t_exp)))
        P_exp = self.energy_to_photon_pow(f_exp * flux_unit)
        print(P_exp.unit)
        w_hat = np.exp(x_hat) * u.Angstrom
        if t_exp is None:
            t_exp = np.zeros(snrs.shape) * u.min
            if hasattr(wavelength_trigger,'__iter__'):

                for i,snr in enumerate(snrs):
                    inds_1 = (w_hat[i,:] < np.max(wavelength_trigger))
                    inds_2 = (w_hat[i,:] > np.min(wavelength_trigger))
                    inds   = (inds_1 * inds_2).astype(bool)
                    t_exp[i] = self.trigger(P_exp[i,inds],snrs[i],w_hat[i,inds])
            else:
                wt_index = np.abs(w_hat - wavelength_trigger).argmin()
                for i,snr in enumerate(snrs):
                    t_exp[i] = self.trigger([P_exp[i,wt_index]],snrs[i],[w_hat[i,wt_index]])
        data['data']['t_exp'] = t_exp

        n_exp = np.empty(P_exp.shape)
        snr_grid = np.empty(P_exp.shape)
        for i in range(P_exp.shape[0]):
            for j in range(P_exp.shape[1]):
                n_exp[i,j] = self.shots(t_exp[i],P_exp[i,j],w_hat[i,j])
                snr_grid[i,j] = self.signal_to_noise(t_exp[i],P_exp[i,j],w_hat[i,j])

        # print('generating true signal to noise ratios...')
        # snr_grid = self.signal_to_noise(t_exp,P_exp)
        print('adding noise...')
        out_shape = snr_grid.shape
        n_readout = self.add_noise(n_exp.flatten(),snr_grid.flatten()).reshape(out_shape)

        data['parameters']['true_snr'] = snr_grid
        data['data']['flux_expected'] = n_exp
        data['data']['flux'] = n_readout

        data['data']['mask'] += (n_readout <= 0.0)
        data['data']['mask'] = data['data']['mask'].astype(bool)
        data['data']['wave'] = self.wave_grid

        # Get Error Bars
        ###################################################
        print('generating exp signal to noise ratios...')
        snr_readout = np.empty(snr_grid.shape)
        for i in range(P_exp.shape[0]):
            for j in range(P_exp.shape[1]):
                snr_readout[i,j] = self.signal_to_noise(t_exp[i],P_exp[i,j],w_hat[i,j],shots=n_readout[i,j])
        print('t_exp: {}\nsnr: {}'.format(np.mean(t_exp),np.mean(snr_readout)))

        print('generating errors...')
        nerr_out = self.generate_errors(n_readout.flatten(),snr_readout.flatten()).reshape(out_shape)

        data['data']['snr_readout'] = snr_readout
        data['data']['ferr']        = nerr_out

        # mask_2 = np.where(np.isnan(nerr_out))

        # Pack Parameters into Dictionary
        ###################################################
        data['parameters']['star'] = {}
        data['parameters']['star'] = dict_of_attr(data['parameters']['star'],self.stellar_model)

        data['parameters']['detector'] = {}
        data['parameters']['detector'] = dict_of_attr(data['parameters']['detector'],self)

        for model in self.transmission_models:
            data['parameters'][model._name] = {}
            data['parameters'][model._name] = dict_of_attr(data['parameters'][model._name],model)
        print('done.')
        return data

    def convolve(self,xs,fs,res):
        '''
            Convolves the total flux with the line spread function, called at .simulate
            Parameters:
            xs: np.ndarray (m) log wavelength array
            fs: np.ndarray (n,m) flux array
            res: res __call__ at log wavelength returns sigma of the gaussian
            Returns:
            f_lsf: np.ndarray (n,m) convolved flux array
        '''        
        def gaussian(x,sigma):
            '''
                Gaussian function that is used to convolve the flux with the line spread function.
                Parameters:
                x: np.ndarray (m) log wavelength array
                sigma: float standard deviation of the gaussian
            '''
            return np.exp(-0.5 * (x/sigma)**2) / (sigma * np.sqrt(2 * np.pi))
        def convolve_element(x,xs,fs):
            '''
                Convolve the flux with the line spread function across element.
                Parameters:
                x: np.ndarray (m) log wavelength array
                fs: np.ndarray (n,m) flux array
            '''
            sigma = res(x)
            kern = gaussian(x - xs,sigma)
            return fs*kern/np.sum(kern)
        
        def convolve_epochs(xs,fs):
            '''
                Convolve the flux with the line spread function across epochs.
                Parameters:
                xs: np.ndarray (m) log wavelength array
                fs: np.ndarray (n,m) flux array
            '''
            return jax.vmap(convolve_element,in_axes=(0,None,None))(xs,xs,fs)
        
        f_lsf = jax.vmap(convolve_epochs,in_axes=(None,0))(xs,fs)
        return f_lsf

    def interpolate_grid(self,xs,x,f):
        '''
            This function takes in the new grid xs and the x and flux arrays output
            by the TheoryModels then interpolates them. If you want to write in
            your own interpolation. Just note that all values coming in are 2d.
            sometimes the first layer is a list because the TheoryModel spit out
            different shapes of flux depending on internal parameters.

            Parameters:
            xs: new grid to interpolate to. 2D ij. i: epoch dimension, j: pixel dimension
        '''
        fs = np.zeros(xs.shape)
        print(xs.shape)
        for i in range(len(x)):
            fs[i,:] = interp.CubicSpline(x[i],f[i])(xs[i,:])
        return fs

    def interpolate_data(self,xs,x,f,dx):
        '''
            Interpolation function that interpolates observing wavelengths onto
            the theoretical flux. Here I use Lanczos interpolation. Defined in lanczos.py.
        '''
        f_exp = jax.vmap(jnp.interp, in_axes=(0,None,0))(xs,x,f)
        # f_exp = jax.vmap(simulacra.lanczos.lanczos_interpolation, in_axes=(0,None,0,None,None))(xs,x,f,dx,self.a)
        return f_exp

    def wave_transform(self,x,epoches,*arg):
        '''
            Wave transformation in detector. Called in .simulate
        '''
        return np.repeat(x[np.newaxis,:],repeats=epoches,axis=0), None

    def trigger(self,P,snr,wavelength):
        '''
            Triggers the detector to stop exposure at a given SNR limit. Only called
            in .simulate if snrs are given not t_exp
        '''
        def func(t,powers,waves):
            out = []
            for i,w in enumerate(waves):
                out.append(self.signal_to_noise(complex(t,0) * u.min, powers[i], waves[i]))#             out = self.signal_to_noise(complex(args[0],0) * u.min, *args[1:])
            return np.abs(np.mean(out) - snr)**2

        res = scipy.optimize.minimize(func, 1.0, args=(P, wavelength))

        return res.x[0] * u.min

    def signal_to_noise(self,t_exp,P,wavelength,shots=None):
        '''
            Calculate signal to noise ratio.
        '''
        if shots is None:
            shots = complex(self.shots(t_exp,P,wavelength),0)
        snr =  shots \
        / (np.sqrt(shots + self.noise_source(t_exp,P,wavelength)))
        return snr

    def shots(self,t_exp,P,wavelength):
        '''
            Convert photons per minute to total photons
        '''
        return t_exp * P

    def noise_source(self,t_exp,P,wavelength):
        '''
            Noise sources of detector
        '''
        return 0.0

    def energy_to_photon_pow(self,flux,*args):
        '''
            convert energy incoming to detector to photons per minute
        '''
        P = self.through_put * (self.area/(const.hbar * const.c) * \
            np.einsum('ij,j,j->ij', flux, self.wave_difference, self.wave_grid)).to(1/u.min)
        return P


class NoisyDetector(Detector):
    def __init__(self,dark_current,read_noise,ccd_eff,**kwargs):
        super(self,NoisyDetector).__init__(kwargs)
        self.dark_current = dark_current
        self.read_noise = read_noise
        self.ccd_eff = ccd_eff

    def shots(self,t_exp,P,wavelength):
        return t_exp * P * self.ccd_eff(wavelength)

    def noise_source(self,t_exp,P,wavelength):
        return self.read_noise(wavelength) + self.dark_current(wavelength) * t_exp

    def ccd_eff():
        doc = "The ccd_eff property."
        def fget(self):
            return self._ccd_eff
        def fset(self, value):
            if hasattr(value,'__call__'):
                self._ccd_eff = value
        def fdel(self):
            del self._ccd_eff
        return locals()
    ccd_eff = property(**ccd_eff())

    def dark_current():
        doc = "The dark_current property."
        def fget(self):
            return self._dark_current
        def fset(self, value):
            if hasattr(value,'__call__'):
                self._dark_current = value
        def fdel(self):
            del self._dark_current
        return locals()
    dark_current = property(**dark_current())

    def read_noise():
        doc = "The read_noise property."
        def fget(self):

            return self._read_noise
        def fset(self, value):

            if hasattr(value,'__call__'):
                self._read_noise = value
        def fdel(self):
            del self._read_noise
        return locals()
    read_noise = property(**read_noise())

class LinearTransformDetector(Detector):
    def __init__(self,w,epsilon,**kwargs):
        super(self,LinearTransformDetector).__init__(kwargs)
        self.epsilon = epsilon
        self.w       = w

    def wave_transform(self,x):
        epoches = xs.shape[0]
        parameters = {'m': np.random.uniform(1-epsilon,1+epsilon,epoches), 'delt': np.random.uniform(0,w,epoches)}
        width = average_difference(out[0,:]) * w
        return x


class JaxDetector(Detector):
    import scipy.sparse
    def convolve(self,xs,fs,dx):
        pass
        # nevermind

def even_wave_grid(wave_min,wave_max,resolution):
    delta_x = simulacra.star.delta_x(4*resolution)
    x_grid = np.arange(np.log(wave_min.to(u.Angstrom).value),np.log(wave_max.to(u.Angstrom).value),delta_x)
    wave_grid = np.exp(x_grid) * u.Angstrom
    return wave_grid

apogee_dict = {'resolution':22_500.0,
            'area': np.pi * (2.5*u.m/2)**2,
            'dark_current': 100/u.s,
            'read_noise': 100,
            'ccd_eff':0.99,
            'through_put':0.05,
            'wave_grid':even_wave_grid(1.51*u.um,1.70*u.um,22_500.0),
            'loc':coord.EarthLocation.of_site('APO')}

keck_dict = {'resolution':100_000.0,
            'area': np.pi * (10*u.m/2)**2,
            'dark_current': 100/u.s,
            'read_noise': 100,
            'ccd_eff':0.99,
            'through_put':0.05,
            'wave_grid':even_wave_grid(500*u.nm,630*u.nm,100_000.0),
            'loc':coord.EarthLocation.of_site('Keck Observatory')}

expres_dict = {'resolution':130_000.0,
            'area': np.pi * (4.3*u.m/2)**2,
            'dark_current': 100/u.s,
            'read_noise': 100,
            'ccd_eff':0.99,
            'through_put':0.05,
            'wave_grid':even_wave_grid(700*u.nm,950*u.nm,130_000.0),
            'loc':coord.EarthLocation.of_site('Lowell Observatory')}

# apogee_det = Detector(**apogee_dict)
# apogee_det.add_model(simulacra.tellurics.TelFitModel(loc=apogee_det.loc,lambmin=apogee_det.lambmin,lambmax=apogee_det.lambmax))
#
# keck_det = Detector(**keck_dict)
# keck_det.add_model(simulacra.tellurics.TelFitModel(loc=keck_det.loc,lambmin=keck_det.lambmin,lambmax=keck_det.lambmax))
# keck_det.add_model(simulacra.gascell.GasCellModel())
#
# expres_det = Detector(**expres_dict)
# expres_det.add_model(simulacra.tellurics.TelFitModel(loc=expres_det.loc,lambmin=expres_det.lambmin,lambmax=expres_det.lambmax))
