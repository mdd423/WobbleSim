import numpy as np
import astropy.units as u
import astropy.time as at
import scipy.interpolate as interp
import scipy.ndimage as img
import scipy.sparse
import numpy.random as random

def dict_from_h5(hf,data):

    import h5py
    for key in hf.keys():
        if isinstance(hf[key], h5py.Group):
            data[key] = {}
            data[key] = dict_from_h5(hf[key],data[key])
        elif key == 'obs_times':
            try:
                print(hf[key])
                data[key] = at.Time(np.array(hf[key]).tolist(),format='isot')
            except TypeError:
                pass
        elif key == 'value':
            return np.array(hf['value']) * u.Unit(hf['unit'][0])
        elif len(hf[key].shape) == 0:
            data[key] = hf[key]
        else:
            data[key] = np.array(hf[key])
    return data

def from_h5(filename):
    import h5py
    data = {}
    with h5py.File(filename,'r') as file:
        data = dict_from_h5(file,data)

    return DetectorData(data)

def convert_xy(x,y,yerr=None,units=u.Angstrom):
    if yerr is not None:
        outerr = yerr / y
    return np.log(x.to(units).value), np.log(y), outerr

def save_dict_as_h5(hf,data):
    import h5py
    for key in data.keys():
        if isinstance(data[key],u.Quantity):
            print('quantity')
            group = hf.create_group(key)
            group.create_dataset('value',data=data[key].value)
            dt = h5py.special_dtype(vlen=str)
            unt = np.array([str(data[key].unit)],dtype=dt)
            group.create_dataset('unit',data=unt)
        elif isinstance(data[key], np.ndarray):
            if data[key].dtype == at.Time:
                print('saving time')
                dt = h5py.special_dtype(vlen=str)
                times = np.array([x.strftime('%Y-%m-%dT%H:%M:%S.%f%z') for x in data[key]],dtype=dt)
                hf.create_dataset(key,data=times)
            else:
                hf.create_dataset(key,data=data[key])
        # except AttributeError:
        elif isinstance(data[key], dict):
            group = hf.create_group(key)
            save_dict_as_h5(group,data[key])
        else:
            print(key, ' saving as string')
            dt = h5py.special_dtype(vlen=str)
            arr = np.array([str(data[key])],dtype=dt)
            hf.create_dataset(key,data=arr)

def from_pickle(filename):
    with open(filename, 'rb') as input:  # Overwrites any existing file.
        model = pickle.load(input)
        return model

class DetectorData:
    def __init__(self,data={}):
        self.data = data

    def to_h5(self,filename):
        import h5py
        hf = h5py.File(filename,"w")
        save_dict_as_h5(hf,self.data)
        hf.close()

    def to_pickle(self,filename):
        import pickle
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def to_fits(self,filename):
        import astropy.io.fits as fits

        hdul = fits.HDUList([])
        for key in self.keys():
            print(key)
            for subkey in self[key].keys():
                print('\t',subkey)
                image_hdu = fits.ImageHDU(self[key][subkey])
                hdul.append(image_hdu)

        hdul.writeto(filename)

    def keys(self):
        return self.data.keys()

    def __getitem__(self,key):
        return self.data[key]

    def __setitem__(self,key,value):
        self.data[key] = value

    def plot_data(self,ax,i,xy=True,units=u.Angstrom,normalize=None,nargs=[]):
        y = self['data']['flux'][i,:]
        x = self['data']['wave']
        yerr = self['data']['ferr'][i,:]
        if normalize is not None:
            y = normalize(y,*nargs)
        if xy:
            x, y, yerr = convert_xy(x, y, yerr, units=units)
        else:
            x = x.value
        ax.errorbar(x,y,yerr,xerr=None,fmt='.k',alpha=0.9,label='Data')
        return ax

    def plot_theory(self,ax,i,xy=True,units=u.Angstrom,normalize=None,nargs=[]):
        y = self['theory']['flux_total'][i,:]
        x = np.exp(self['theory']['x_theory']) * u.Angstrom
        if normalize is not None:
            y = normalize(y,*nargs)
        if xy:
            x, y, _ = convert_xy(x, y, None, units=units)
        else:
            x= x.value
        ax.plot(x, y,'.',color='gray',alpha=0.4,label='Total ' + self.__class__.__name__,markersize=3)
        return ax

    def plot_lsf(self,ax,i,xy=True,units=u.Angstrom,normalize=None,nargs=[]):
        y = self['theory']['flux_lsf'][i,:]
        x = np.exp(self['theory']['x_theory']) * u.Angstrom
        if normalize is not None:
            y = normalize(y,*nargs)
        if xy:
            x, y, _ = convert_xy(x, y, None, units=units)
        else:
            x = x.value
        ax.plot(x, y,'.',color='magenta',alpha=0.4,label='LSF ' + self.__class__.__name__,markersize=3)
        return ax

    def plot_star(self,ax,i,xy=True,units=u.Angstrom,normalize=None,nargs=[]):
        y = self['theory']['flux_star'][i,:]
        x = np.exp(self['theory']['x_theory']) * u.Angstrom
        if normalize is not None:
            y = normalize(y,*nargs)
        if xy:
            x, y, _ = convert_xy(x, y, None, units=units)
        else:
            x = x.value
        ax.plot(x, y,'.',color='red',alpha=0.4,label='LSF ' + self.__class__.__name__,markersize=3)
        return ax

    def plot_flux(self,ax,i,flux_keys,wave_keys,pargs=[],ferr_keys=None,xy=True,units=u.Angstrom,normalize=None,nargs=[]):
        y = self[flux_keys][i,:]
        x = self[wave_keys]
        if ferr_keys is not None:
            yerr = self[ferr_keys][i,:]
        if normalize is not None:
            y = normalize(y,*nargs)
        if xy:
            x, y, _ = convert_xy(x, y, None, units=units)
        else:
            x = x.value
        if ferr_keys is None:
            ax.plot(x, y,*pargs)
        else:
            ax.errorbar(x, y, yerr,*pargs)
        return ax

    def plot_rvs(self,ax,units=u.km/u.s):
        now = at.Time.now()
        ax.plot(self['data']['obs_times'] - now,self['parameters']['rvs'].to(units).value,'.k')
        return ax

    def plot_rvs_minus_bcs(self,ax,units=u.km/u.s):
        now = at.Time.now()
        bcs = simulacra.star.get_berv(self['data']['obs_times'],self['parameters']['obs'],
                                        self['parameters']['ra'],self['parameters']['dec'],
                                        self['parameters']['velocity_drift'])
        ax.plot(self['data']['times'] - now,(self['parameters']['rvs'] - bcs).to(units).value,'.k')
        return ax

    def plot_rvs_minus_bcs_mod_period(self,ax,units=u.km/u.s):
        now = at.Time.now()
        bcs = simulacra.star.get_berv(self['data']['obs_times'],self['parameters']['obs'],
                                        self['parameters']['ra'],self['parameters']['dec'],
                                        self['parameters']['velocity_drift'])
        ax.plot((self['data']['times'] - now) % self['parameters']['period'],(self['parameters']['rvs'] - bcs).to(units).value,'.k')
        return ax
