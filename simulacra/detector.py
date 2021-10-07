import numpy as np
import astropy.units as u
import scipy.interpolate as interp
import scipy.ndimage as img
import scipy.sparse
import numpy.random as random

def interpolate(x,xs,ys):
    spline = interp.CubicSpline(xs,ys)
    return spline(x)

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

def get_s2n(shape,constant):
    return np.ones(shape) * constant

def hermitegaussian(coeffs,x,sigma):
    xhat = (x/sigma)
    herms = np.polynomial.hermite.Hermite(coeffs)
    return herms(xhat) * np.exp(-xhat**2)


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
        f_out[jj] = np.dot(f_in[max(0,jj-centering):min(size,jj+n-centering)], kernel[max(0,centering-jj):min(n,size-jj+centering)])
    return f_out

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

def average_difference(x):
    return np.mean([t - s for s, t in zip(x, x[1:])])


class DetectorData:
    def __init__(self,data):
        self.data = data
        self.wave = data['data']['wave']
        self.flux = data['data']['flux']
        self.ferr = data['data']['ferr']

    def to_h5(self,filename):
        import h5py

        hf = h5py.File(filename,"w")
        print(self.data.keys())
        for key in self.data.keys():
            group = hf.create_group(key)
            print(key)
            # print(self.data[key])
            for subkey in self.data[key].keys():
                print('\t'+subkey, type(self.data[key][subkey]))
                if subkey == 'times':
                    dt = h5py.special_dtype(vlen=str)
<<<<<<< HEAD
                    times = np.array([x.strftime("%d-%b-%Y (%H:%M:%S.%f)") for x in self.data[key][subkey]],dtype=dt)
=======
                    times = np.array([x.strftime('%Y-%m-%dT%H:%M:%S.%f%z') for x in self.data[key][subkey]],dtype=dt)
>>>>>>> dd69062d333c9ee4f1d7fd9fa9628e1f9fb07908
                    print(times)
                    group.create_dataset(subkey,data=times)
                else:
                    try:
                        group.create_dataset(subkey,data=self.data[key][subkey])
                    except TypeError:
                        print(subkey, ' saving as string')
                        dt = h5py.special_dtype(vlen=str)
                        arr = np.array([str(self.data[key][subkey])],dtype=dt)
                        group.create_dataset(subkey,data=arr)

        hf.close()

    def __getidex__(self,key):
        return self.data[key]

    @property
    def x(self):
        return np.log(self.wave/u.Angstrom)

    @property
    def y(self):
        return np.log(self.flux)

    @property
    def yerr(self):
        return self.ferr/self.flux

    def plot_xy(self,ax,epoch_idx,normalize=None,nargs=[]):
        y = self.y[epoch_idx,:]
        if normalize is not None:
            y = normalize(y,*nargs)
        ax.errorbar(self.x,y,self.yerr[epoch_idx,:],xerr=None,fmt='.k',alpha=0.9,label='Data')
        return ax

    def plot_wf(self,xs,epoch_idx,noralize=None,nargs=[]):
        y = self.flux[epoch_idx,:]
        if normalize is not None:
            y = normalize(y,*nargs)
        ax.errorbar(self.wave,y,self.ferr[epoch_idx,:],xerr=None,fmt='.k',alpha=0.9,label='Data')
        return ax

    def plot_the(self,ax,epoch_idx,normalize=None,nargs=[]):
        y = self.data['theory']['flux_the'][epoch_idx,:]
        if normalize is not None:
            y = normalize(y,*nargs)
        ax.plot(self.data['theory']['wave_the'], y,'.',color='gray',alpha=0.4,label='Total ' + self.__class__.__name__,markersize=3)
        return ax

    def plot_lsf(self,ax,epoch_idx,normalize=None,nargs=[]):
        y = self.data['theory']['flux_lsf'][epoch_idx,:]
        if normalize is not None:
            y = normalize(y,*nargs)
        ax.plot(self.data['theory']['wave_the'], y,'.',color='magenta',alpha=0.4,label='LSF ' + self.__class__.__name__,markersize=3)
        return ax


class Detector:
    def __init__(self,stellar_model,resolution,epsilon=0.0,s2n=20,gamma=1.0,w=0.0,a=4):
        '''Detector model that simulates spectra from star given resolution...

        Detector takes on a given theoretical `stellar_model`, `resolution`, with
        a constant signal to noise ratio, `s2n`. Then call simulate to generate a given number
        of epoches of spectral data.

        Parameters
        ----------
        stellar_model: TheoryModel with a generate_data method that returns flux, wave, and deltas
        resolution: the resolution of the detector being simulated
        epsilon: a small that is used to pull the random stretchs for the wavelength grid, m ~ uniform(1-epsilon,1+epsilon)
        s2n: the constant signal to noise ratio
        gamma: a constant that random effects the errorbars of flux \sigma_f ~ normal(mu=0.0,sigma=gamma * f_i,j/s2n_i,j)
        w: a constant that random affects the jitter to the wavelength grid, del ~ uniform(-w*pxl_width, w*pxl_width)
        a: the number of kernel functions used at the lanczos interpolation step

        Returns
        -------
        data: DetectorData type that contains all parameters generate
        '''

        self.stellar_model = stellar_model
        self.transmission_models = []

        self.epsilon = epsilon
        self.s2n     = s2n
        self.gamma   = gamma
        self.w       = w
        self.a       = a

        self.resolution = resolution

        self._lambmin = 0.0 * u.nm
        self._lambmax = 100000 * u.nm

        self.sigma_range   = 5.0
        self.lsf_const_coeffs = [1.0]
        self.sigma      = 1.0/resolution


    def add_model(self,model):
        self.transmission_models.append(model)
        if self.lambmin < model.lambmin:
            self.lambmin = model.lambmin
        if self.lambmax > model.lambmax:
            self.lambmax = model.lambmax

    def checkmin(self):
        minimums = [self.stellar_model.lambmin]
        for model in self.transmission_models:
            minimums += [model.lambmin]
        # print(minimums)
        return max(minimums)

    @property
    def lambmin(self):
        value = self.checkmin()
        # print(value)
        if  value <= self._lambmin:
            return self._lambmin
        else:
            self._lambmin = value
            print('lambmin to low for all grids\nsetting to lambmin to {}'.format(value))
            return value
        # assert self._lambmin > max(minimums)

    @lambmin.setter
    def lambmin(self,lambmin):
        value = self.checkmin()
        if value <= lambmin:
            self._lambmin = lambmin
        else:
            self._lambmin = value
            print('lambmin to low for all grids\nsetting to lambmin to {}'.format(value))

    def checkmax(self):
        maximums = [self.stellar_model.lambmax]
        for model in self.transmission_models:
            maximums += [model.lambmax]
        # print(maximums)
        return min(maximums)

    @property
    def lambmax(self):
        value = self.checkmax()
        # print(value)
        if value >= self._lambmax:
            return self._lambmax
        else:
            self._lambmax = value
            print('lambmax to high for all grids\nsetting to lambmax to {}'.format(value))
            return value

    @lambmax.setter
    def lambmax(self,lambmax):
        value = self.checkmax()
        if value >= lambmax:
            self._lambmax = lambmax
        else:
            self._lambmax = value
            print('lambmax to high for all grids\nsetting to lambmax to {}'.format(value))

    @property
    def xmin(self):
        return np.log(self._lambmin/u.Angstrom)

    @property
    def xmax(self):
        return np.log(self._lambmax/u.Angstrom)

    def simulate(self,epoches,convolve_on=True):

        flux, wave, deltas = self.stellar_model.generate_spectra(epoches)
        differences = [get_median_difference(self.stellar_model.x)]
        print('generating spectra...')
        for model in self.transmission_models:
            model.generate_transmission(epoches)
            differences += [get_median_difference(model.x[iii,:]) for iii in range(model.x.shape[0])]
            print(model, differences)
        new_step_size = min(differences)
        print(new_step_size)
        # Initialize interpolating arrays
        ###################################################################
        self.xs = np.arange(self.xmin,self.xmax,step=new_step_size)
        self.stellar_model.fs = np.empty((epoches,self.xs.shape[0]))
        self.stellar_model.xs = self.xs
        for model in self.transmission_models:
            model.xs = self.xs
            model.fs = np.empty((epoches,self.xs.shape[0]))

        # Interpolate all models and combine onto detector
        ##################################################################
        fs = np.empty((epoches,self.xs.shape[0]))
        print('interpolating spline...')
        for i in range(epoches):
            print(i)
            self.stellar_model.fs[i,:] = interpolate(self.xs + deltas[i],self.stellar_model.x,self.stellar_model.flux)
            fs[i,:] = self.stellar_model.fs[i,:]
            for model in self.transmission_models:
                model.fs[i,:] = interpolate(self.xs,model.x[i,:],model.flux[i,:])
                fs[i,:] *= model.fs[i,:]
        # loop through stellar, tellurics, and gas cell models and generate spectra
        # stellar model needs to exist at the very least
        # interpolate, convolve, jitter, stretch, noise, signal to noise ratio, errorbars
        self.lsf_coeffs = np.outer(np.ones((fs.shape[1],len(self.lsf_const_coeffs))), self.lsf_const_coeffs)

        # should be an array that can vary over pixel j or hermite m
<<<<<<< HEAD
        sigma = 1.0/self.resolution
        self.lsf_centering = 'centered'
=======
        self.lsf_centering = 'centered'
        f_lsf = np.empty(fs.shape)
>>>>>>> dd69062d333c9ee4f1d7fd9fa9628e1f9fb07908
        print('convolving...')
        if convolve_on:
            for ii in range(f_lsf.shape[0]):
                f_lsf[ii,:] = convolve_hermites(fs[ii,:],self.lsf_coeffs,self.lsf_centering,self.sigma,self.sigma_range,new_step_size)
        else:
            f_lsf = fs

        # Generate dataset grid & jitter & stretch
        ##################################################
        res_step_size = spacing_from_res(self.resolution)
        x = np.arange(self.xmin,self.xmax,step=res_step_size)
        x_hat, m    = stretch(x,epoches,self.epsilon)
        x_hat, delt = jitter(x,epoches,self.w)

        # Interpolate Spline and Add Noise
        ##################################################
        s2n_grid  = get_s2n(x_hat.shape,self.s2n)
        f_exp     = np.empty(x_hat.shape)
        f_readout = np.empty(x_hat.shape)
        print('interpolating lanczos...')
        for i in range(f_exp.shape[0]):
            print(i)
            f_exp[i,:] = lanczos_interpolation(x_hat[i,:],self.xs,f_lsf[i,:],dx=new_step_size,a=self.a)
            for j in range(f_exp.shape[1]):
                f_readout[i,j] = f_exp[i,j] * random.normal(1,1./s2n_grid[i,j])

        # Get Error Bars
        ###################################################
        ferr_out = generate_errors(f_readout,s2n_grid,self.gamma)

        # Pack Output into Dictionary
        ###################################################
                # detector data
        data = {"data":
                {"wave":np.exp(x) * u.Angstrom,
                "wave_unit":u.Angstrom,
                "flux":f_readout,
                "flux_exp":f_exp,
                "ferr":ferr_out,
                "times":self.stellar_model.times},
                # parameters of both star and telescope
                "parameters":
                {"del":delt,
                "m":m,
                "a":self.a,
                "lsf_coeffs":self.lsf_coeffs,
                "ra":self.stellar_model.ra,
                "ra_unit":self.stellar_model.ra.unit,
                "dec":self.stellar_model.dec,
                "dec_unit":self.stellar_model.dec.unit,
                "obs":self.stellar_model.observatory_name,
                "rvs":self.stellar_model.rvs,
                "rv_unit":self.stellar_model.rvs.unit,
                "period":self.stellar_model.period,
                "period_unit":self.stellar_model.period.unit,
                "resolution":self.resolution},
                # detector and transmission theoretical model
                "theory":
                {"wave_the":np.exp(self.xs)*u.Angstrom,
                "wave_the_unit": u.Angstrom,
                "flux_lsf":f_lsf,
                "flux_the":fs,
                "flux_star":self.stellar_model.fs}
                }
        # transmission models parameters and theory
        for model in self.transmission_models:
            try:
                for key in model.parameters.keys():
                    data['parameters'][key] = model.parameters[key]
            except AttributeError:
                pass
            data['theory']["trans " + model.__class__.__name__] = model.fs
        out = DetectorData(data)

        return out
