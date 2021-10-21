from simulacra.theory import TheoryModel
import scipy.io
import astropy.units as u
import numpy as np

def read_in_idl(filename='data/gascell/keck_fts_renorm.idl'):
    array = scipy.io.readsav(filename)
    transmission = array['siod']
    wavelength   = array['wiod']
    return transmission, wavelength

class GasCellModel(TheoryModel):
    def __init__(self,filename='../data/gascell/keck_fts_renorm.idl'):
        self._name = 'gascell'
        self.filename = filename
        transmission, wavelength = read_in_idl(self.filename)
        self.flux = transmission
        self.wave = wavelength * u.Angstrom

    def generate_transmission(self,star,detector,obs_times,exp_times):
        self.flux = np.repeat(np.expand_dims(self.flux,axis=0),obs_times.shape[0],axis=0)
        self.wave = np.repeat(np.expand_dims(self.wave,axis=0),obs_times.shape[0],axis=0)
        return self.flux, self.wave
