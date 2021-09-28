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

def convolve_hermites(f_in,coeffs,centering,x,sigma):
    f_out = np.empty(f_in.shape)
    size = f_in.shape[1]
    n    = x.shape[0]
    for ii in range(f_out.shape[0]):
        for jj in range(f_out.shape[1]):
            kernel = hermitegaussian(coeffs[jj,:],x,sigma)
            # L1 normalize the kernel so the total flux is conserved
            kernel /= np.sum(kernel)
            f_out[ii,jj] = np.dot(f_in[ii,max(0,jj-centering):min(size,jj+n-centering)], kernel[max(0,centering-jj):min(n,size-jj+centering)])
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

# def mean_lsf(low_resolution,spacing,sigma_range=5.0):
#     lsf = np.arange(-sigma_range/low_resolution,sigma_range/low_resolution,step=spacing)
#     lsf = gauss_func(lsf,mu=0.0,sigma=1.0/low_resolution)
#     lsf /= np.linalg.norm(lsf,ord=1)
#     return lsf
#
# def gauss_func(x,mu,sigma):
#     return np.exp((-1/2)*(x - mu)**2/(sigma**2))

def average_difference(x):
    return np.mean([t - s for s, t in zip(x, x[1:])])


class DetectorData:
    def __init__(self,wave,flux,ferr):
        self.wave = wave
        self.flux = flux
        self.ferr = ferr

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


class Detector:
    def __init__(self,stellar_model,resolution,epsilon=0.001,s2n=20,gamma=1.0,w=1.0,a=4):
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

    def simulate(self,epoches):

        flux, wave, deltas = self.stellar_model.generate_spectra(epoches)
        differences = [get_median_difference(self.stellar_model.x)]
        for model in self.transmission_models:
            model.generate_transmission(epoches)
            differences += [get_median_difference(model.x[iii,:]) for iii in range(model.x.shape[0])]
        new_step_size = min(differences)

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
        self.fs = np.empty((epoches,self.xs.shape[0]))
        for i in range(epoches):
            self.stellar_model.fs[i,:] = interpolate(self.xs + deltas[i],self.stellar_model.x,self.stellar_model.flux)
            self.fs[i,:] = self.stellar_model.fs[i,:]
            for model in self.transmission_models:
                model.fs[i,:] = interpolate(self.xs,model.x[i,:],model.flux[i,:])
                self.fs[i,:] *= model.fs[i,:]
        # loop through stellar, tellurics, and gas cell models and generate spectra
        # stellar model needs to exist at the very least
        # interpolate, convolve, jitter, stretch, noise, signal to noise ratio, errorbars
        self.lsf_coeffs      = np.ones((self.fs.shape[1],2))
        self.lsf_coeffs[:,1] *= 0.25

        sigma = 1.0/self.resolution
        x          = np.arange(-self.sigma_range * sigma,self.sigma_range * sigma,step=new_step_size)
        self.lsf_centering = int(x.shape[0]/2)
        self.f_lsf = convolve_hermites(self.fs,self.lsf_coeffs,self.lsf_centering,x,sigma) #np.einsum('jm,im->ij',mat,self.fs)

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
        for i in range(f_exp.shape[0]):
            f_exp[i,:] = lanczos_interpolation(x_hat[i,:],self.xs,self.f_lsf[i,:],dx=new_step_size,a=self.a)
            for j in range(f_exp.shape[1]):
                f_readout[i,j] = f_exp[i,j] * random.normal(1,1./s2n_grid[i,j])

        # Get Error Bars
        ###################################################
        ferr_out = generate_errors(f_readout,s2n_grid,self.gamma)
        lmb_out  = np.exp(x)

        data = DetectorData(lmb_out,f_readout,ferr_out)

        # Pack Output into Dictionary
        ###################################################
        out = {"wavelength_sample":lmb_out,
                "flux":f_readout,
                "flux_error":ferr_out,
                "wavelength_theory":np.exp(self.xs),
                "flux_lsf":self.f_lsf,
                "del":delt,
                "m":m}
        out["flux " + self.stellar_model.__class__.__name__] = self.stellar_model.fs
        for model in self.transmission_models:
            out["flux " + model.__class__.__name__] = model.fs

        return out, data

    def to_h5(self):
        import h5py

        hf = h5py.File(out_name,"w")
        theory_group = hf.create_group("theory")
        theory_group.create_dataset("wavelength",data=out["wavelength_theory"])
        theory_group.create_dataset("flux_stellar",data=out["flux_stellar"])
        theory_group.create_dataset("flux_tellurics",data=out["flux_tellurics"])
        theory_group.create_dataset("flux_gas",data=out["flux_gas"])
        theory_group.create_dataset("flux_lsf",data=out["flux_lsf"])

        sample_group = hf.create_group("samples")
        sample_group.create_dataset("wavelength",data=out["wavelength_sample"])
        sample_group.create_dataset("flux",data=out["flux"])
        sample_group.create_dataset("flux_error",data=out["flux_error"])

        consts_group = hf.create_group("constants")
        consts_group.create_dataset("m",data=out["m"])
        consts_group.create_dataset("del",data=out["del"])
        consts_group.create_dataset("airmass",data=out["airmass"])
        consts_group.create_dataset("delta",data=out["delta"])
        consts_group.create_dataset("low_resolution",data=args.lr)

        hf.close()

    def plot(self,ax,epoch_idx,normalize=None,nargs=[]):
        y = self.fs[epoch_idx,:]
        if normalize is not None:
            y = normalize(y,*nargs)
        ax.plot(self.xs, y,'.',color='gray',alpha=0.4,label='Total ' + self.__class__.__name__,markersize=3)
        return ax

    def plot_lsf(self,ax,epoch_idx,normalize=None,nargs=[]):
        y = self.f_lsf[epoch_idx,:]
        if normalize is not None:
            y = normalize(y,*nargs)
        ax.plot(self.xs, y,'.',color='magenta',alpha=0.4,label='LSF ' + self.__class__.__name__,markersize=3)
        return ax
