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

def download_phoenix_model(star,outdir=None):
    directories = ['data','HiResFITS','PHOENIX-ACES-AGSS-COND-2011','Z{:+.1f}'.format(star.z)]
    # print(directories)
    if star.alpha != 0.0:
        directories[-1] += '.Alpha={:+.2f}'.format(star.alpha)

    filename = 'lte{:05d}-{:1.2f}'.format(star.temperature,star.logg) + directories[-1][1:] + '.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
    outname = os.path.join(outdir,filename)
    print(outname)
    if os.path.isfile(outname):
        print('using saved flux file')
        return outname

    else:
        from ftplib import FTP

        ftp = FTP('phoenix.astro.physik.uni-goettingen.de') #logs in
        ftp.login()
    #     for directory in directories:
        print("ftp: {}\nfilename: {}\ndirs: {}".format(ftp, filename,directories))
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

def get_berv(times,obs,ra,dec,v_d):
    obj = coord.SkyCoord(ra,dec,radial_velocity=v_d)
    loc = coord.EarthLocation.of_site(obs)
    bc  = obj.radial_velocity_correction(obstime=times,location=loc).to(u.km/u.s)
    return bc

def binary_system_velocity(times,amplitude,period,phase_time='2000-01-02'):
    starttime = atime.Time(phase_time)
    ptime = (times - starttime)/period
    return amplitude * np.sin(2*np.pi*ptime*u.radian)

def get_velocity_measurements(times,amplitude,period,loc,target):
    # berv  = get_berv(times,obs,ra,dec,velocity_drift)
    berv = target.radial_velocity_correction(obstime=times,location=loc)
    rvs  = berv + binary_system_velocity(times,amplitude,period)
    return rvs

def get_night_grid(loc,tstart,tend,steps_per_night=10):
    from astroplan import Observer
    observer = Observer(location=loc)
    days = int((tend - tstart) / u.day)

    # print(days)
    all_times = tstart + np.linspace(0,days,days + 1)
    sunset  = observer.sun_set_time(all_times,which='next')
    sunrise = observer.sun_rise_time(all_times,which='next')

    # print(all_times)
    outarr = np.array([])
    for i in range(len(sunrise)-1):
        nighttime = (sunrise[i+1]-sunset[i])
        outarr = np.concatenate((outarr,
                                sunset[i] + \
                                np.linspace(0,nighttime.value,steps_per_night)))
    return outarr

def get_realistic_times(target,loc,all_times):

    telescope_frame = coord.AltAz(obstime=all_times,location=loc)
    secz = np.array(target.transform_to(telescope_frame).secz)
    isvis = secz > 0

    # time? alltimes
    possible_times = all_times[isvis]
    airmass = secz[isvis]

    isreasonable = airmass < 3.0
    return possible_times[isreasonable], airmass[isreasonable]


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

def stellar_to_detector_flux(star,detector,exp_times):
    stellar_area = 4. * np.pi * star.stellar_radius**2
    ratio_of_areas = detector.area / (4.* np.pi * star.distance**2)

    print("photon flux: {:.2e}".format(np.mean(star.wave_difference * star.surface_flux.to(u.photon/u.s / u.m**3, u.spectral_density(star.wave))).to(u.ph/u.m**2/u.s).value))
    print("ratios: {:.2e}".format(ratio_of_areas.to(1).value))
    print("exposures: {:.2e}".format(exp_times[0].to(u.s).value))
    print("star area: {:.2e}".format(stellar_area.to(u.m**2).value))
    det_flux = detector.through_put * np.outer(exp_times, np.multiply(star.surface_flux.to(u.photon/u.s / u.m**3, \
                            u.spectral_density(star.wave)) \
                            , star.wave_difference)) * stellar_area * ratio_of_areas
    print("det flux: {:.2e}".format(np.mean(det_flux).to(u.ph).value))
    return det_flux.to(u.ph)

class PhoenixModel(TheoryModel):
    def __init__(self,distance,alpha,z,temperature,logg,target,amplitude,period,outdir=None):
        super(PhoenixModel,self).__init__()
        if outdir is None:
            self.outdir = os.path.join('..','data','stellar','PHOENIX')
            os.makedirs(self.outdir,exist_ok=True)
        self.temperature = temperature
        self.z = z
        self.logg  = logg
        self.alpha = alpha
        self.wavename = download_phoenix_wave(self.outdir)
        self.fluxname = download_phoenix_model(self,self.outdir)

        grid = astropy.io.fits.open(self.fluxname)
        self.stellar_radius = grid['PRIMARY'].header['PHXREFF'] * u.cm
        self.surface_flux = grid['PRIMARY'].data * u.erg / u.cm**3 / u.s
        self.wave     = read_in_fits(self.wavename) * u.Angstrom

        # make these attributes of the phoenix model
        self.distance = coord.Distance(distance)
        self.target = target
        self.amplitude = amplitude
        self.period    = period

    def wave_difference():
        doc = "The wave_difference property."
        def fget(self):
            diff = 0.1 * u.Angstrom * np.ones(self.wave.shape)
            return diff
        return locals()
    wave_difference = property(**wave_difference())

    def generate_spectra(self,detector,obs_times,exp_times):
        # add integral over transmission
        time = atime.Time([obs_times[i] + exp_times[i]/2 for i in range(len(obs_times))])
        rvs    = get_velocity_measurements(time,self.amplitude,self.period,detector.loc,self.target)
        deltas = shifts(rvs)

        print('surface flux: mean {:3.2e}\t median {:3.2e}'.format(np.mean(self.surface_flux),np.median(self.surface_flux)))
        obs_flux = self.surface_flux * (self.stellar_radius**2/self.distance**2).to(1)
        print('obs     flux: mean {:3.2e}\t median {:3.2e}'.format(np.mean(obs_flux),np.median(obs_flux)))
        # axes.plot(self.wave,obs_flux,'or',alpha=0.3)
        # # axes.set_xlim(6120,6130)
        # plt.show()
        obs_flux = np.outer(np.ones(obs_times.shape),obs_flux)
        # obs_flux = stellar_to_detector_flux(self,detector,exp_times)
        return obs_flux, self.wave, deltas, rvs

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
