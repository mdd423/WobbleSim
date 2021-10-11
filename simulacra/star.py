import numpy as np

import astropy.units as u
import astropy.constants as const
import astropy.io.fits
import astropy.time as atime
import astropy.coordinates as coord

import numpy.random as random
import os

from simulacra.theory import TheoryModel

def sample_deltas(epoches,vel_width=30*u.km/u.s):
    deltas  = np.array(shifts((2*random.rand(epoches)-1)*vel_width))
    return deltas

def read_in_fits(filename):
    print('reading in {}'.format(filename))
    grid = astropy.io.fits.open(filename)['PRIMARY'].data
    # flux_all    = astropy.io.fits.open(fluxfile)['PRIMARY'].data
    return grid

def download_phoenix_wave(outdir):
    filename = 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'
    outname = os.path.join(outdir,filename)
    if os.path.isfile(outname):
        print('using saved wave file')
        return outname
    else:
        from ftplib import FTP

        ftp = FTP('phoenix.astro.physik.uni-goettingen.de') #logs in
        ftp.login()
        ftp.cwd('HiResFITS')
        ftp.retrlines('LIST')

        with open(outname, 'wb') as fp:
            ftp.retrbinary('RETR ' + filename, fp.write) # start downloading

        ftp.close() # close the connection

        return outname

def download_phoenix_model(alpha,z,temperature,logg,outdir=None):
    directories = ['HiResFITS','PHOENIX-ACES-AGSS-COND-2011','Z{:+.1f}'.format(z)]
    # print(directories)
    if alpha != 0.0:
        directories[-1] += '.Alpha={:+.2f}'.format(alpha)

    filename = 'lte{:05d}-{:1.2f}'.format(temperature,logg) + directories[-1][1:] + '.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
    outname = os.path.join(outdir,filename)
    if os.path.isfile(outname):
        print('using saved wave file')
        return outname

    else:
        from ftplib import FTP

        ftp = FTP('phoenix.astro.physik.uni-goettingen.de') #logs in
        ftp.login()
    #     for directory in directories:
        ftp.cwd(os.path.join(*directories))
        ftp.retrlines('LIST')


        with open(outname, 'wb') as fp:
            ftp.retrbinary('RETR ' + filename, fp.write) # start downloading

        ftp.close() # close the connection

        return outname

def zplusone(vel):
    return np.sqrt((1 + vel/(const.c))/(1 - vel/(const.c)))

def shifts(vel):
    return np.log(zplusone(vel))

def get_random_times(n,tframe=365*u.day):
    now = atime.Time.now()
    dts = np.random.uniform(0,tframe.value,n) * tframe.unit
    times = now + dts
    return times

def get_berv(times,observatory_name,ra,dec,velocity_drift):
    obj = coord.SkyCoord(ra,dec,radial_velocity=velocity_drift)
    loc = coord.EarthLocation.of_site(observatory_name)
    bc  = obj.radial_velocity_correction(obstime=times,location=loc).to(u.km/u.s)
    return bc

def binary_system_velocity(times,amplitude,period,phase_time='2000-01-02'):
    starttime = atime.Time(phase_time)
    ptime = (times - starttime)/period
    return amplitude * np.sin(2*np.pi*ptime*u.radian)

def get_velocity_measurements(times,amplitude,observatory_name,ra,dec,period,epoches,velocity_drift):
    berv  = get_berv(times,observatory_name,ra,dec,velocity_drift)

    rvs   = berv + binary_system_velocity(times,amplitude,period)
    return rvs


class StarModel(TheoryModel):
    def __init__(self,deltas):
        # super(StarModel,self).__init__()
        self._deltas = deltas

    @property
    def deltas(self):
        return self._deltas

    @deltas.setter
    def deltas(self,deltas):
        self._deltas = deltas


class PhoenixModel(TheoryModel):
    def __init__(self,alpha,z,temperature,logg,ra,dec,observatory_name,amplitude,period,velocity_drift,outdir=None):
        super(PhoenixModel,self).__init__()
        if outdir is None:
            self.outdir = os.path.join('data','stellar','PHOENIX')
            os.makedirs(self.outdir,exist_ok=True)
        self.temperature = temperature
        self.z     = z
        self.logg  = logg
        self.alpha = alpha
        self.wavename = download_phoenix_wave(self.outdir)
        self.wave     = read_in_fits(self.wavename) * u.Angstrom
        self.color = 'red'
        # make these attributes of the phoenix model
        self.observatory_name = observatory_name
        self.ra, self.dec = ra, dec
        # amplitude = np.random.uniform(a_min.value,a_max.to(a.unit).value) * a_min.unit
        self.amplitude = amplitude
        self.period    = period
        self.velocity_drift = velocity_drift

    def generate_spectra(self,epoches,times):
        self.times = times
        self.rvs    = get_velocity_measurements(self.times,self.amplitude,self.observatory_name,self.ra,self.dec,self.period,epoches,self.velocity_drift)
        self.deltas = shifts(self.rvs)
        # self.deltas = sample_deltas(epoches,self.velocity_padding)
        fluxname = download_phoenix_model(self.alpha,self.z,self.temperature,self.logg,self.outdir)
        self.flux = read_in_fits(fluxname)

        return self.flux, self.wave, self.deltas

    def plot(self,ax,epoch_idx,normalize=None,nargs=[]):
        y = self.flux
        if normalize is not None:
            y = normalize(y,*nargs)
        ax.plot(self.x - self.deltas[epoch_idx],y,'o',color=self.color,alpha=0.4,label='Truth ' + self.__class__.__name__,markersize=4)
        return ax

    def plot_interpolated(self,ax,epoch_idx,normalize=None,nargs=[]):
        # import matplotlib.pyplot as plt
        y = self.fs[epoch_idx,:]
        if normalize is not None:
            y = normalize(y,*nargs)
        ax.plot(self.xs,y,'.',color=self.color,alpha=0.3,label='Interpolated ' + self.__class__.__name__,markersize=3)
        return ax
