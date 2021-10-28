import numpy as np
import astropy.units as u
import astropy.constants as const
import astropy.time as at
import scipy.interpolate as interp
import scipy.ndimage as img
import scipy.sparse
import numpy.random as random
import logging

from simulacra.dataset import DetectorData

from itertools import repeat
from multiprocessing import Pool

def get_median_difference(x):

    return np.median([t - s for s, t in zip(x, x[1:])])

def spacing_from_res(R):
    return np.log(1+1/R)

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
    x = np.arange(-sigma_range * sigma,sigma_range * sigma,step=spacing)

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
        kernel = hermitegaussian(coeffs[jj,:],x,sigma)
        # L1 normalize the kernel so the total flux is conserved
        kernel /= np.sum(kernel)
        f_out[jj] = np.dot(f_in[max(0,jj-centering):min(size,jj+n-centering)]\
                                    ,kernel[max(0,centering-jj):min(n,size-jj+centering)])
    return f_out

def interpolate(x,xs,ys):
    spline = interp.CubicSpline(xs,ys)
    return spline(x)

def lanczos_interpolation(x,xs,ys,dx,a=4):
    x0 = xs[0]
    y = np.zeros(x.shape)
    for i,x_value in enumerate(x):
        # which is basically the same as sample=x[j-a+1] to x[j+a]
        # where j in this case is the nearest index xs_j to x_value
        sample_min,sample_max = max(0,abs(x_value-x0)//dx - a + 1),min(xs.shape[0],abs(x_value-x0)//dx + a)
        samples = np.arange(sample_min,sample_max,dtype=int)
        for sample in samples:
            y[i] += ys[sample] * lanczos_kernel((x_value - xs[sample])/dx,a)
    return y

def lanczos_kernel(x,a):
    if x == 0:
        return 1
    if x > -a and x < a:
        return a*np.sin(np.pi*u.radian*x) * np.sin(np.pi*u.radian*x/a)/(np.pi**2 * x**2)
    return 0

def generate_errors(f,snr):
    xs,ys = np.where(f < 0)
    for x,y in zip(xs,ys):
        f[x,y] = 0
    f_err = np.empty(f.shape)
    for i in range(f_err.shape[0]):
        for j in range(f_err.shape[1]):
            f_err[i,j] = f[i,j]/snr[i,j]
    return f_err

def average_difference(x):
    return np.mean([t - s for s, t in zip(x, x[1:])])

def add_noise(f_exp,snr_grid):
    f_readout = np.empty(f_exp.shape)
    for i in range(f_exp.shape[0]):
        print('snr {}: {}'.format(i,np.median(snr_grid[i,:])))
        for j in range(f_exp.shape[1]):
            f_readout[i,j] = f_exp[i,j] + random.normal(0.0,1./snr_grid[i,j])
    return f_readout

def signal_to_noise_ratio(detector,flux,exp_times):
    xs,ys = np.where(flux < 0)
    for x,y in zip(xs,ys):
        flux[x,y] = 0

    snr = np.empty(flux.shape)
    for i in range(snr.shape[0]):
        for j in range(snr.shape[1]):
            # if j % 500 == 0:
            #     print(detector.read_noise[j], (detector.dark_current[j] * exp_times[i]).to(1), detector.ccd_eff[j] * flux[i,j])
            # implicitly flux is already dependent on exposure time
            snr[i,j] = detector.ccd_eff[j] * flux[i,j] \
            / np.sqrt(detector.read_noise[j] + detector.dark_current[j] * exp_times[i] + detector.ccd_eff[j] * flux[i,j])
    return snr

def get_masks(xs,mask_the,x_hat):
    return np.array([interp.interp1d(xs,mask_the[i,:].astype(float),kind='nearest')(x_hat[i,:]) for i in range(x_hat.shape[0])]).astype(bool)


class Detector:
    def __init__(self,stellar_model,resolution,loc,area,wave_grid,dark_current,read_noise,ccd_eff,through_put=0.2,wave_padding=5*u.Angstrom,epsilon=0.0,gamma=1.0,w=0.0,a=4):
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

        # Randomized Parameters
        self.epsilon = epsilon
        self.gamma   = gamma
        self.w       = w
        self.a       = a

        # Simulation Parameters
        self._lambmin = 0.0 * u.nm
        self._lambmax = 100000 * u.nm

        # Detector Properties
        self.wave_grid    = wave_grid
        self.wave_padding = wave_padding
        self.dark_current = dark_current
        self.read_noise   = read_noise
        self.ccd_eff      = ccd_eff
        self.through_put  = through_put
        self.area = area
        self.loc  = loc
        # LSF properties
        self.sigma_range   = 5.0
        self.resolution   = resolution
        self.lsf_const_coeffs = [1.0]
        self.sigma      = 1.0/resolution

        self.trans_cutoff = 10.

    # def resolution():
    #     doc = "The resolution property."
    #     def fget(self):
    #         return self._resolution
    #     def fset(self, value):
    #         if len(value) == 1:
    #             self._resolution = value
    #         if value.shape == self.wave_grid.shape:
    #             self._resolution = value
    #         else:
    #             logging.error('resolution grid needs to be the same \
    #                             shape as the wave_grid')
    #         del self.sigma
    #         del self.kernel
    #     def fdel(self):
    #         logging.warn('overwriting resolution')
    #         del self._resolution
    #     return locals()
    # resolution = property(**resolution())
    #
    # def sigma():
    #     doc = "The sigma property."
    #     def fget(self):
    #         return self._sigma
    #     def fset(self, value):
    #         self._sigma = value
    #         del self.kernel
    #         del self.resolution
    #     def fdel(self):
    #         logging.warn('overwriting resolution')
    #         del self._sigma
    #     return locals()
    # sigma = property(**sigma())
    #
    # def kernel():
    #     doc = "The kernel property."
    #     def fget(self):
    #         return self._kernel
    #     def fset(self, value):
    #         self._kernel = value
    #         del self.resolution
    #         del self.sigma
    #     def fdel(self):
    #         logging.warn('overwriting kernel')
    #         del self._kernel
    #     return locals()
    # kernel = property(**kernel())
    #
    def transmission():
        doc = "The transmission property."
        def fget(self):
            return self._transmission
        def fset(self, value):
            self._transmission = value
        def fdel(self):
            del self._transmission
        return locals()
    transmission = property(**transmission())

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
                print("wave_grid min -> {}".format(maximum))
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

    def ccd_eff():
        doc = "The ccd_eff property."
        def fget(self):
            return self._ccd_eff
        def fset(self, value):
            try:
                temp = iter(value)
                self._ccd_eff = value
            except TypeError:
                self._ccd_eff = value * np.ones(self.wave_grid.shape)
        def fdel(self):
            del self._ccd_eff
        return locals()
    ccd_eff = property(**ccd_eff())

    def dark_current():
        doc = "The dark_current property."
        def fget(self):
            return self._dark_current
        def fset(self, value):
            try:
                temp = iter(value)
                self._dark_current = value
            except TypeError:
                self._dark_current = value * np.ones(self.wave_grid.shape)
        def fdel(self):
            del self._dark_current
        return locals()
    dark_current = property(**dark_current())

    def read_noise():
        doc = "The read_noise property."
        def fget(self):
            return self._read_noise
        def fset(self, value):
            try:
                temp = iter(value)
                self._read_noise = value
            except TypeError:
                self._read_noise = value * np.ones(self.wave_grid.shape)
        def fdel(self):
            del self._read_noise
        return locals()
    read_noise = property(**read_noise())

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

    def simulate(self,obs_times,exp_times,convolve_on=True):
        data = DetectorData()
        data['data'] = {}
        data['data']['obs_times'] = obs_times
        data['data']['exp_times'] = exp_times
        data['data']['epoches']   = obs_times.shape[0]
        epoches = obs_times.shape[0]
        data['theory'] = {}
        data['theory']['star'] = {}
        flux_stellar, wave_stellar, deltas, rvs = self.stellar_model.generate_spectra(self,obs_times,exp_times)
        print('right after generate spectra: ', np.where(~np.isfinite(flux_stellar)))
        data['data']['rvs'], data['theory']['star']['deltas'] = rvs, deltas
        data['theory']['star']['flux'], data['theory']['star']['wave'] = flux_stellar, wave_stellar
        differences = [get_median_difference(np.log(wave_stellar.to(u.Angstrom).value))]
        print('generating spectra...')
        trans_flux, trans_wave = [], []
        data['theory']['interpolated'] = {}
        for model in self.transmission_models:
            data['theory']['interpolated'][model._name] = {}
            data['theory'][model._name] = {}
            flux, wave = model.generate_transmission(self.stellar_model,self,obs_times,exp_times)
            trans_flux.append(flux), trans_wave.append(wave)
            differences += [get_median_difference(np.log(wave[iii][:].to(u.Angstrom).value)) for iii in range(len(wave))]
            data['theory'][model._name]['flux'],data['theory'][model._name]['wave'] = flux, wave
            print(model, differences)
        new_step_size = min(differences)

        # Interpolate all models and combine onto detector
        # PARALLELIZE
        ##################################################################
        data['theory']['interpolated']['star'] = {}
        xs = np.arange(np.log(self.lambmin.to(u.Angstrom).value),np.log(self.lambmax.to(u.Angstrom).value),step=new_step_size)
        print('interpolating spline...')
        stellar_arr = np.empty((epoches,xs.shape[0]))
        trans_arrs  = np.empty((len(self.transmission_models),epoches,xs.shape[0]))
        for i in range(epoches):
            print(i)
            stellar_arr[i,:] = interpolate(xs + deltas[i],np.log(wave_stellar.to(u.Angstrom).value),flux_stellar[i,:].to(u.erg/u.s/u.cm**3).value)
            for j,model in enumerate(self.transmission_models):
                trans_arrs[j,i,:] = interpolate(xs,np.log(trans_wave[j][i][:].to(u.Angstrom).value),trans_flux[j][i][:])
        print('combining grids...')
        data['theory']['interpolated']['star']['flux'] = stellar_arr
        fs = stellar_arr.copy()
        flux_unit = flux_stellar.unit
        mask_the = np.zeros(fs.shape,dtype=bool)
        for j,model in enumerate(self.transmission_models):
            # print("fs: ", fs.shape)
            fs *= trans_arrs[j,:,:]
            mask_the = (trans_arrs[j,:,:] > self.trans_cutoff) | mask_the
            data['theory']['interpolated'][model._name]['flux'] = trans_arrs[j,:,:]
        data['theory']['interpolated']['total'] = {}
        data['theory']['interpolated']['total']['flux'] = fs
        data['theory']['interpolated']['total']['wave'] = np.exp(xs) * u.Angstrom
        data['theory']['interpolated']['total']['mask'] = mask_the

        # Convolving using Hermite Coeffs
        #################################################
        self.lsf_coeffs = np.outer(np.ones((fs.shape[1],len(self.lsf_const_coeffs))), self.lsf_const_coeffs)
        # should be an array that can vary over pixel j or hermite m
        self.lsf_centering = 'centered'
        f_lsf = np.empty(fs.shape)
        print('convolving...')
        class ConvolveIter:
            def __init__(obj,fs):
                obj.fs = fs
                obj._i =  0

            def __iter__(obj):
                # output = (fs[obj._i,:], self.lsf_coeffs,self.lsf_centering,self.sigma,self.sigma_range,new_step_size)
                obj._i = 0
                return obj

            def __next__(obj):

                if obj._i == fs.shape[0]:
                    raise StopIteration
                print(obj._i)
                output = (fs[obj._i,:], self.lsf_coeffs,self.lsf_centering,self.sigma,self.sigma_range,new_step_size)
                obj._i += 1
                return output

        with Pool() as pool:
            obj = ConvolveIter(fs)
            M = pool.starmap(convolve_hermites, obj)
            print(M,len(M))

        f_lsf = np.asarray(M)

        data['theory']['lsf'] = {}
        data['theory']['lsf']['flux'] = f_lsf

        # Generate transform wavelength grid using jitter & stretch
        ##################################################
        x           = np.log(self.wave_grid.to(u.Angstrom).value)#np.arange(self.xmin,self.xmax,step=res_step_size)
        x_hat, m    = stretch(x,epoches,self.epsilon)
        x_hat, delt = jitter(x,epoches,self.w)
        data['parameters'] = {}
        data['parameters']['wavetransform'] = {}
        data['parameters']['wavetransform']['m'] = m
        data['parameters']['wavetransform']['delt'] = delt

        print('xs: {} {}\nxhat: {} {}'.format(np.exp(np.min(xs)),np.exp(np.max(xs)),np.exp(np.min(x_hat)),np.exp(np.max(x_hat))))
        data_mask = get_masks(xs,mask_the,x_hat)
        data['data']['mask'] = data_mask

        # Interpolate using Lanczos and Add Noise
        # PARALLELIZE
        ##################################################
        f_exp     = np.empty(x_hat.shape)
        print('interpolating lanczos...')

        class LanczosIter:
            def __init__(obj):

                obj._i =  0

            def __iter__(obj):
                # output = (fs[obj._i,:], self.lsf_coeffs,self.lsf_centering,self.sigma,self.sigma_range,new_step_size)
                obj._i = 0
                return obj

            def __next__(obj):
                if obj._i == f_lsf.shape[0]:
                    raise StopIteration
                print(obj._i)
                output = (x_hat[obj._i,:],xs,f_lsf[obj._i,:],new_step_size,self.a)
                obj._i += 1
                return output

        with Pool() as pool:
            obj = LanczosIter()
            M = pool.starmap(lanczos_interpolation, obj)
            print(M,len(M))

        f_exp = np.asarray(M)

        # for i in range(f_exp.shape[0]):
        #     f_exp[i,:] = lanczos_interpolation(x_hat[i,:],xs,f_lsf[i,:],dx=new_step_size,a=self.a)
        #     print('f_exp {} median: {} {}'.format(i,np.median(f_exp[i,:]),flux_unit))

        # print(f_exp.unit)
        print('area: {}\t avg d lambda: {}\t avg lambda: {}\t avg exp times: {}'.format(self.area,np.mean(self.wave_difference),np.mean(self.wave_grid),np.mean(exp_times)))
        n_exp = self.through_put * (self.area/(const.hbar * const.c)*np.einsum('ij,j,j,i->ij',f_exp * flux_unit,self.wave_difference,self.wave_grid,exp_times)).to(1)
        for i in range(n_exp.shape[0]):
            print('{} n mean: {:3.2e}\t n median: {:3.2e}'.format(i,np.mean(n_exp[i,~data_mask[i,:]]),np.median(n_exp[i,~data_mask[i,:]])))

        print('generating signal to noise ratios...')
        snr_grid = signal_to_noise_ratio(self,n_exp,exp_times)
        print('adding noise...')
        n_readout = add_noise(n_exp,snr_grid)

        data['data']['snr'] = snr_grid
        data['data']['flux_expected'] = n_exp
        data['data']['flux'] = n_readout
        data['data']['wave'] = self.wave_grid

        # Get Error Bars
        # PARALLELIZE
        ###################################################
        nerr_out = generate_errors(n_readout,snr_grid)
        data['data']['ferr'] = nerr_out

        # Pack Parameters into Dictionary
        ###################################################
        data['parameters']['star'] = {}
        data['parameters']['star'] = dict_of_attr(data['parameters']['star'],self.stellar_model)

        data['parameters']['detector'] = {}
        data['parameters']['detector'] = dict_of_attr(data['parameters']['detector'],self)

        for model in self.transmission_models:
            data['parameters'][model._name] = {}
            data['parameters'][model._name] = dict_of_attr(data['parameters'][model._name],model)
        return data

def dict_of_attr(data,obj):
    obj_list = [a for a in dir(obj) if not a.startswith('__')]
    for ele in obj_list:
        try:
            data[ele] = getattr(obj, ele)
        except AttributeError:
            pass
    return data
