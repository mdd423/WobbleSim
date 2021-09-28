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
    def __init__(self,filename):
        self.filename = filename
        transmission, wavelength = read_in_idl(self.filename)
        self.flux = transmission
        self.wave = wavelength   * u.Angstrom
        self.color = 'green'

    def generate_transmission(self,epoches):
        self.flux = np.repeat(np.expand_dims(self.flux,axis=0),epoches,axis=0)
        self.wave = np.repeat(np.expand_dims(self.wave,axis=0),epoches,axis=0)
        # print('gas flux',self.flux.shape)
        # print('gas wave',self.wave.shape)
        return self.flux, self.wave
