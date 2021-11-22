#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0,'/home/mdd423/WobbleSim')
print(sys.path)
import simulacra.star
import simulacra.tellurics
from simulacra.star import PhoenixModel

import random
random.seed(a=102102102)
import numpy as np
np.random.seed(102102102)

import astropy.io.fits
import astropy.time as at

import astropy.units as u
import astropy.coordinates as coord
import astropy.constants as const


# <h1>11 - Making Some APOGEE Data</h1>
# In this test, we are going to generate data from the same star at multiple distances using the same conditions as apogee.

# In[2]:


from datetime import datetime
import os
if __name__ == '__main__':
    date = datetime.today().strftime('%Y-%m-%d')
    outdir = os.path.join('/scratch/mdd423/simulacra/out/',date)
    os.makedirs(outdir,exist_ok=True)


    # <h2>Detector</h2>
    # Here we define our detector giving it an aperature area, resolution, dark current, read noise, and ccd efficiency. All of these can be except area can be given as an array of the same size as the wave_grid (eg. if the detector has varying resolution or noise levels)

    # In[3]:


    import simulacra.detector
    det_dict = simulacra.detector.expres_dict


    # In[4]:


    loc = det_dict['loc']
    ra, dec = np.random.uniform(0,360) * u.degree, np.random.uniform(loc.lat.to(u.degree).value-30,loc.lat.to(u.degree).value+30) * u.degree
    target = coord.SkyCoord(ra,dec,frame='icrs')


    # In[5]:


    tstart = at.Time('2020-01-01T08:10:00.123456789',format='isot',scale='utc')
    tend   = tstart + 720 * u.day
    night_grid = simulacra.star.get_night_grid(loc,tstart,tend,steps_per_night=20)
    possible_times, airmass = simulacra.star.get_realistic_times(target,loc,night_grid)


    # In[6]:


    epoches = 30


    # Now we selected some random sample of these to observe at and the airmasses at those times

    # In[7]:


    obs_ints = random.sample(range(len(airmass)),epoches)
    obs_times, obs_airmass = possible_times[obs_ints], airmass[obs_ints]


    # In[8]:


    # import matplotlib.pyplot as plt
    import scipy.ndimage
    def normalize(y,yerr,sigma):
        y_low = scipy.ndimage.gaussian_filter(y,sigma)
        return y/y_low, yerr/y


    # In[9]:


    exp_time = 8 * u.minute

    exp_times = np.ones(epoches)*exp_time


    # <h2>Simulations</h2>
    # Now we are going to simulate this star with the same detector defined by the above parameters at many different distances.

    # In[10]:


    distances = [100 * u.pc, 150 * u.pc, 200 * u.pc, 250 * u.pc, 300 * u.pc, 350 * u.pc, 400 * u.pc, 450 * u.pc, 500 * u.pc, 550 * u.pc, 750 * u.pc, 1 * u.kpc]


    # In[11]:


    len(distances)


    # Now Simulate! And plot outputs before saving!

    # In[12]:


    logg = 1.0
    T    = 4800
    z    = -1.0
    alpha= 0.4
    amplitude = 4 * u.km/u.s
    period    = 67.3 * u.day


    # In[ ]:


    sigma = 200
    n_plots = 4
    plt_unit = u.Angstrom
    sort_times = np.argsort(obs_times)

    # fig, axes = plt.subplots(len(distances),figsize=(10 * len(distances),10 * n_plots),sharex=True,sharey=True)
    # fig.text(0.5, 0.04, 'Wavelength [{}]'.format(plt_unit), ha='center', va='center')
    # fig.text(0.06, 0.5, 'Flux', ha='center', va='center', rotation='vertical')

    # fig_rv, ax_rvs = plt.subplots(len(distances),2,figsize=(20 * len(distances),10 * 2))

    for i,distance in enumerate(distances):
        stellar_model = PhoenixModel(distance,alpha,z,T,logg,target,amplitude,period)


        detector = simulacra.detector.Detector(stellar_model,**det_dict)

        tellurics_model = simulacra.tellurics.TelFitModel(detector.lambmin,detector.lambmax,loc)
        detector.add_model(tellurics_model)
        data = detector.simulate(obs_times,exp_times)

        filename = os.path.join(outdir,'expres_e{}_d{}_R{}_a{}_p{}_l{:3.1e}{:3.1e}_snr{:2.1e}'.format(epoches,distance.to(u.pc).value,det_dict['resolution'],                                                                amplitude.to(u.m/u.s).value,                                                                period.to(u.day).value,                                                                detector.lambmin.value,                                                                detector.lambmax.value,                                                                np.mean(data['data']['snr_readout'][~np.isnan(data['data']['snr_readout'])])))
        print(filename)
        data.to_h5(filename + '.h5')
        # Defining and plotting flux from star on detector
        j = 1
        print('{:3.2e}'.format(np.mean(data['data']['flux'][j,:])),'{:3.2e}'.format(np.mean(data['data']['ferr'][j,:])))
    #     flux, ferr = normalize(data['data']['flux'][j,:],data['data']['ferr'][j,:],sigma)
    #     axes[i].errorbar(np.log(data['data']['wave'].to(u.Angstrom).value),flux,yerr=ferr,fmt='.k',alpha=0.5)
    #     #     data.plot_data(axes[i],sort_times[i],xy='x',units=plt_unit)
    #     #     data.plot_tellurics(axes[i],sort_times[i],xy='x',units=plt_unit)#,normalize=normalize,nargs=[sigma]
    #     #     data.plot_gas(axes[i],sort_times[i],xy='x',units=plt_unit)
    #     #     data.plot_theory(axes[i],sort_times[i],xy='x',units=plt_unit)
    #     #     data.plot_lsf(axes[i],sort_times[i],xy='x',units=plt_unit)
    #     #     data.plot_star(axes[i],sort_times[i],xy='x',units=plt_unit)
    #     # plt.savefig('out/datatest5.png')

    #     times = at.Time([obs_times[k] + exp_times[k]/2 for k in range(len(obs_times))])
    #     rv = data['data']['rvs'].to(u.km/u.s)
    #     bc = target.radial_velocity_correction(obstime=times,location=loc).to(u.km/u.s)
    #     eprv = rv - bc

    #     v_unit = u.m/u.s
    # #     ax_rvs[i,0].set_ylim(-2.1,2.1)
    #     ax_rvs[i,0].plot((times - min(times)).to(u.day).value % period.to(u.day).value,eprv.to(v_unit).value,'.r')
    #     ax_rvs[i,0].set_xlabel('time [d]')
    #     ax_rvs[i,0].set_ylabel('vel [{}]'.format(v_unit))

    #     v_unit = u.km/u.s
    # #     ax_rvs[i,1].set_ylim(-35,35)
    #     ax_rvs[i,1].plot((times - min(times)).to(u.day).value,rv.to(v_unit).value,'.k')
    #     ax_rvs[i,1].set_xlabel('time [d]')
    #     ax_rvs[i,1].set_ylabel('vel [{}]'.format(v_unit))
    # plt.show()


    # In[ ]:
