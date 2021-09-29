import numpy as np
import astropy.units as u


class TheoryModel:
    def __init__(self):
        self._flux = np.array([])
        self._wave = np.array([])

    @property
    def wave(self):
        return self._wave

    @wave.setter
    def wave(self,wave):
        self._lambmin = np.min(wave)
        self._lambmax = np.max(wave)
        self._wave = wave

    @property
    def lambmin(self):
        return self._lambmin

    @property
    def lambmax(self):
        return self._lambmax

    @property
    def x(self):
        return np.log(self._wave/u.Angstrom)

    def plot(self,ax,epoch_idx,normalize=None,nargs=[]):
        import matplotlib.pyplot as plt
        y = self.flux[epoch_idx,:]
        if normalize is not None:
            y = normalize(y,*nargs)
        ax.plot(self.x[epoch_idx,:], y,'o',color=self.color,alpha=0.4,label='Truth ' + self.__class__.__name__,markersize=4)
        return ax

    def plot_interpolated(self,ax,epoch_idx,normalize=None,nargs=[]):
        import matplotlib.pyplot as plt
        # import matplotlib.pyplot as plt
        y = self.fs[epoch_idx,:]
        if normalize is not None:
            y = normalize(y,*nargs)
        ax.plot(self.xs, y,'.',color=self.color,alpha=0.3,label='Interpolated ' + self.__class__.__name__,markersize=3)
        return ax
