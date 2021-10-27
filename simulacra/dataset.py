import numpy as np
import astropy.units as u
import astropy.time as at
import astropy.coordinates as coord
import scipy.interpolate as interp
import scipy.ndimage as img
import scipy.sparse
import numpy.random as random

def dict_from_h5(hf,data):

    import h5py
    for key in hf.keys():
        if key == 'obs_times':
            try:
                print(hf[key])
                data[key] = at.Time(np.array(hf[key]).tolist(),format='isot')
            except TypeError:
                pass
        elif key == 'loc':
            # print(hf[key]['value'][()],hf[key]['unit'][()][0])
            data[key] = coord.EarthLocation.from_geocentric(hf['loc']['value'][()][0] * u.Unit(hf['loc']['unit'][()][0]) \
                            ,hf['loc']['value'][()][1] * u.Unit(hf['loc']['unit'][()][0]) \
                            ,hf['loc']['value'][()][2] * u.Unit(hf['loc']['unit'][()][0]))
        elif key == 'target':
            data[key] = coord.SkyCoord(hf['target']['ra'][0] * u.deg,hf['target']['dec'][0] * u.deg)
        elif key == 'value':
            return np.array(hf['value']) * u.Unit(hf['unit'][0])
        elif isinstance(hf[key], h5py.Group):
            data[key] = {}
            data[key] = dict_from_h5(hf[key],data[key])
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
    outerr = None
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
        elif isinstance(data[key], coord.EarthLocation):
            print('saving location...{}'.format(data[key].geocentric))
            hf.create_dataset(key,data[key].geocentric)
        elif isinstance(data[key],coord.SkyCoord):
            print('saving target...{} {}'.format(data[key].ra,data[key].dec))
            group = hf.create_group(key)
            group.create_dataset('ra',data=[data[key].ra.to(u.deg).value])
            group.create_dataset('dec',data=[data[key].dec.to(u.deg).value])
        else:
            print(key, ' saving as string')
            dt = h5py.special_dtype(vlen=str)
            arr = np.array([str(data[key])],dtype=dt)
            hf.create_dataset(key,data=arr)

def from_pickle(filename):
    import pickle
    with open(filename, 'rb') as input:  # Overwrites any existing file.
        model = pickle.load(input)
        return model

data_plot_settings = {'marker':'.','color':'black','alpha':0.9,'label':'Data'}
interpolated_settings = {'marker':'.','alpha':0.3,'label':'Interpolated','markersize':3}
truth_settings = {'marker':'o','alpha':0.4,'label':'Truth','markersize':4}

gas_settings = {'color':'green'}
star_settings = {'color':'red'}
tellurics_settings = {'color':'blue'}
lsf_settings = {'color':'pink'}

def print_keys(data,extra=''):
    for key in data.keys():
        print(extra + '{}: {}'.format(key, type(data[key])))
        if isinstance(data[key],dict):
            tab = '--'
            extra += tab
            print_keys(data[key],extra)
            extra = extra[:-len(tab)]

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

    def plot_flux(self,ax,i,flux_keys,wave_keys,pargs=[],ferr_keys=None,xy='',units=u.Angstrom,normalize=None,nargs=[]):
        y_data = self
        for key in flux_keys:
            y_data = y_data[key]
        if len(y_data.shape) == 1:
            y = y_data
        else:
            y = y_data[i,:]

        x_data = self
        for key in wave_keys:
            x_data = x_data[key]
        if len(x_data.shape) == 1:
            x = x_data
        else:
            x = x_data[i,:]

        yerr = 0.0
        if ferr_keys is not None:
            err_data = self
            for key in ferr_keys:
                err_data = err_data[key]
            yerr = err_data[i,:]

        if normalize is not None:
            y, yerr = normalize(y,yerr,*nargs)
        if 'x' in xy:
            x = np.log(x.to(u.Angstrom).value)
            # x, y, _ = convert_xy(x, y, None, units=units)
        else:
            x = x.value
        if 'y' in xy:
            y, yerr = np.log(y), yerr/y
        if ferr_keys is None:
            ax.plot(x, y, **pargs)
        else:
            ax.errorbar(x, y, yerr,**pargs)
        return ax

    def plot_data(self,ax,i,xy='',units=u.Angstrom,normalize=None,nargs=[]):
        self.plot_flux(ax,i,['data','flux'],['data','wave'],ferr_keys=['data','ferr'],pargs=data_plot_settings,xy=xy,units=units,normalize=normalize,nargs=nargs)
        return ax

    def plot_theory(self,ax,i,xy='',units=u.Angstrom,normalize=None,nargs=[]):
        self.plot_flux(ax,i,['theory','interpolated','total','flux'],['theory','interpolated','total','wave'],xy=xy,pargs={'color':'gray',**interpolated_settings},units=units,normalize=normalize,nargs=nargs)
        return ax

    def plot_lsf(self,ax,i,xy='',units=u.Angstrom,normalize=None,nargs=[]):
        self.plot_flux(ax,i,['theory','lsf','flux'],['theory','interpolated','total','wave'],xy=xy,pargs={'color':'pink',**interpolated_settings},units=units,normalize=normalize,nargs=nargs)
        return ax

    def plot_star(self,ax,i,xy='',units=u.Angstrom,normalize=None,nargs=[]):
        self.plot_flux(ax,i,['theory','interpolated','star','flux'],['theory','interpolated','total','wave'],pargs={**star_settings,**interpolated_settings},xy=xy,units=units,normalize=normalize,nargs=nargs)
        return ax

    def plot_gas(self,ax,i,xy='',units=u.Angstrom,normalize=None,nargs=[]):
        self.plot_flux(ax,i,['theory','interpolated','gascell','flux'],['theory','interpolated','total','wave'],xy=xy,units=units,pargs={**gas_settings,**interpolated_settings},normalize=normalize,nargs=nargs)
        return ax

    def plot_tellurics(self,ax,i,xy='',units=u.Angstrom,normalize=None,nargs=[]):
        self.plot_flux(ax,i,['theory','interpolated','tellurics','flux'],['theory','interpolated','total','wave'],xy=xy,pargs={**tellurics_settings,**interpolated_settings},normalize=normalize,nargs=nargs)
        return ax

    def plot_rvs(self,ax,units=u.km/u.s):
        now = at.Time.now()
        time = np.array([(x - at.Time.now()).value for x in self['data']['obs_times']],dtype=float)
        rvs = self['data']['rvs'].to(units).value
        ax.plot(time,rvs,'.k')
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
