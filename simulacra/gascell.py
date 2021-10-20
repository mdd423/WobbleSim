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
        self.filename = filename
        transmission, wavelength = read_in_idl(self.filename)
        self.flux = transmission
        self.wave = wavelength * u.Angstrom
        self.color = 'green'

    def generate_transmission(self,times):
        self.flux = np.repeat(np.expand_dims(self.flux,axis=0),times.shape[0],axis=0)
        self.wave = np.repeat(np.expand_dims(self.wave,axis=0),times.shape[0],axis=0)
        return self.flux, self.wave
