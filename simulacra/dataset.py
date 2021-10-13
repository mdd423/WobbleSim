import numpy as np
import astropy.units as u
import astropy.time as at
import scipy.interpolate as interp
import scipy.ndimage as img
import scipy.sparse
import numpy.random as random

def from_h5(filename):
    import h5py

    data = {}
    with h5py.File(filename,'r') as file:
        for key in file.keys():
            data[key] = {}
            print('key')
            for subkey in file[key].keys():
                print('\t', subkey)
                if subkey == 'times':
                    data[key][subkey] = at.Time(np.array(file[key][subkey]).tolist(),format='isot')
                else:
                    if isinstance(file[key][subkey],h5py.Group):
                        data[key][subkey] = np.array(file[key][subkey]['value']) * u.Unit(file[key][subkey]['unit'][0])
                    else:
                        if len(file[key][subkey].shape) == 0:
                            data[key][subkey] = file[key][subkey][()]
                        else:
                            data[key][subkey] = np.array(file[key][subkey])
    return DetectorData(data)

def convert_xy(x,y,yerr=None,units=u.Angstrom):
    if yerr is not None:
        outerr = yerr / y
    return np.log(x.to(units).value), np.log(y), outerr


class DetectorData:
    def __init__(self,data):
        self.data = data

    def to_h5(self,filename):
        import h5py

        hf = h5py.File(filename,"w")
        print(self.keys())
        for key in self.keys():
            group = hf.create_group(key)
            print(key)
            for subkey in self[key].keys():
                print('\t'+subkey, type(self[key][subkey]))
                if subkey == 'times':
                    dt = h5py.special_dtype(vlen=str)
                    times = np.array([x.strftime('%Y-%m-%dT%H:%M:%S.%f%z') for x in self[key][subkey]],dtype=dt)
                    group.create_dataset(subkey,data=times)
                else:
                    if isinstance(self[key][subkey],u.Quantity):
                        print('quantity')
                        subgroup = group.create_group(subkey)
                        subgroup.create_dataset('value',data=self[key][subkey].value)
                        dt = h5py.special_dtype(vlen=str)
                        unt = np.array([str(self[key][subkey].unit)],dtype=dt)
                        subgroup.create_dataset('unit',data=unt)
                    else:
                        try:
                            group.create_dataset(subkey,data=self[key][subkey])
                        except TypeError:
                            print(subkey, ' saving as string')
                            dt = h5py.special_dtype(vlen=str)
                            arr = np.array([str(self[key][subkey])],dtype=dt)
                            group.create_dataset(subkey,data=arr)

        hf.close()

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

    def plot_data(self,ax,i,xy=True,units=u.Angstrom,normalize=None,nargs=[]):
        y = self['data']['flux'][i,:]
        x = self['data']['wave']
        yerr = self['data']['ferr'][i,:]
        if normalize is not None:
            y = normalize(y,*nargs)
        if xy:
            x, y, yerr = convert_xy(x, y, yerr, units=units)
        ax.errorbar(x,y,yerr,xerr=None,fmt='.k',alpha=0.9,label='Data')
        return ax

    def plot_theory(self,ax,i,xy=True,units=u.Angstrom,normalize=None,nargs=[]):
        y = self['theory']['flux_the'][i,:]
        x = self['theory']['wave_the']
        if normalize is not None:
            y = normalize(y,*nargs)
        if xy:
            x, y, _ = convert_xy(x, y, None, units=units)
        ax.plot(x, y,'.',color='gray',alpha=0.4,label='Total ' + self.__class__.__name__,markersize=3)
        return ax

    def plot_lsf(self,ax,i,xy=True,units=u.Angstrom,normalize=None,nargs=[]):
        y = self['theory']['flux_lsf'][i,:]
        x = self['theory']['wave_the']
        if normalize is not None:
            y = normalize(y,*nargs)
        if xy:
            x, y, _ = convert_xy(x, y, None, units=units)
        ax.plot(x, y,'.',color='magenta',alpha=0.4,label='LSF ' + self.__class__.__name__,markersize=3)
        return ax

    def plot_star(self,ax,i,xy=True,units=u.Angstrom,normalize=None,nargs=[]):
        y = self['theory']['flux_star'][i,:]
        x = self['theory']['wave_the']
        if normalize is not None:
            y = normalize(y,*nargs)
        if xy:
            x, y, _ = convert_xy(x, y, None, units=units)
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
        if ferr_keys is None:
            ax.plot(x, y,*pargs)
        else:
            ax.errorbar(x, y, yerr,*pargs)
        return ax

    def plot_rvs(self,ax,units=u.km/u.s):
        now = at.Time.now()
        ax.plot(self['data']['times'] - now,self['parameters']['rvs'].to(units).value,'.k')
        return ax

    def plot_rvs_minus_bcs(self,ax,units=u.km/u.s):
        now = at.Time.now()
        bcs = simulacra.star.get_berv(self['data']['times'],self['parameters']['obs'],
                                        self['parameters']['ra'],self['parameters']['dec'],
                                        self['parameters']['velocity_drift'])
        ax.plot(self['data']['times'] - now,(self['parameters']['rvs'] - bcs).to(units).value,'.k')
        return ax

    def plot_rvs_minus_bcs_mod_period(self,ax,units=u.km/u.s):
        now = at.Time.now()
        bcs = simulacra.star.get_berv(self['data']['times'],self['parameters']['obs'],
                                        self['parameters']['ra'],self['parameters']['dec'],
                                        self['parameters']['velocity_drift'])
        ax.plot((self['data']['times'] - now) % self['parameters']['period'],(self['parameters']['rvs'] - bcs).to(units).value,'.k')
        return ax
