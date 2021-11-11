# Standard library
import argparse
import os
import pathlib
import shutil
import sys

import simulacra.star
import simulacra.tellurics
import simulacra.detector
import simulacra.gascell

# Third-party
import numpy as np
# from threadpoolctl import threadpool_limits

# Package
# from .helpers import get_parser
# from ..log import logger
import random
import astropy.coordinates as coord
import astropy.units as u
import astropy.time as at
random.seed(102102102)

def run_simulation(detector,transmission_models,exp_times,epoches,window):
    # parser args into these constants and filename
    tstart = at.Time('2020-01-01T08:10:00.123456789',format='isot',scale='utc')
    tend   = tstart + window * u.day

    night_grid = simulacra.star.get_night_grid(detector.loc,tstart,tend,steps_per_night=5)
    possible_times, airmass = simulacra.star.get_realistic_times(detector.stellar_model.target,detector.loc,night_grid)


    obs_ints = random.sample(range(len(airmass)),epoches)
    obs_times, obs_airmass = possible_times[obs_ints], airmass[obs_ints]
    for model in transmission_models:
        detector.add_model(model)

    data = detector.simulate(obs_times,exp_times)
    return data

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--epoches',type=int,required=True,help='number of epoches of data to generate')

    parser.add_argument('-d','--distance',type=float,required=True,help='distance to star in pcs')
    parser.add_argument('-p','--period',type=float,default=40.3,help='period of the star wobble in days')
    parser.add_argument('-a','--amp',type=float,default=2,help='amplitude of velocity wobble in km/s')
    parser.add_argument('--alpha',type=float,default=0.4,help='The alpha ratio of the star being observed must be in the PHOENIX repository')
    parser.add_argument('-z',type=float,default=-1.0,help='The z ratio of the star being observed must be in the PHOENIX repository')
    parser.add_argument('-T','--temp',type=float,default=4600,help='The temperature in Kelvin of the star being observed must be in the PHOENIX repository')
    parser.add_argument('--logg',type=float,default=1.0,help='The logg of the star being observed must be in the PHOENIX repository')
    parser.add_argument('--amplitude',type=float,default=2.0,help='The amplitude of oscillation of the star in km/s being observed must be in the PHOENIX repository')

    parser.add_argument('--epsilon',type=float,default=1.0,help='random property of the wave transformation')
    parser.add_argument('-w',type=float,default=0.0,       help='random property of the wave transformation')
    parser.add_argument('--gamma',type=float,default=1.0,  help='user set parameter to control SNR')


    parser.add_argument('--ra',type=float,default=None,help='The right ascension of the star being observed if left empty it will be random set.')
    parser.add_argument('--dec',type=float,default=None,help='The right ascension of the star being observed if left empty it will be random set.')

    parser.add_argument('--window',type=float,default=180,help='Time in days to observe the star over')
    parser.add_argument('--exp_time',type=float,default=8,help='Time in minutes of each exposure')
    return parser

def add_tellurics_args(parser):
    parser.add_argument('--pressure',type=float,default=1.0e6,help='The pressure at the observatory at the time of the observations in pascals')
    parser.add_argument('--temperature',type=float,default=300,help='The temperature at the observatory at the time of the observations in Kelvin')
    parser.add_argument('--humidity',type=float,default=50.0,help='The humidity at the observatory at the time of the observations in percentage')

    return parser

def get_star(loc,args):
    if args.ra is None:
        args.ra = np.random.uniform(0,360) * u.degree
    if args.dec is None:
        args.dec = np.random.uniform(loc.lat.to(u.degree).value-30,loc.lat.to(u.degree).value+30) * u.degree
    target = coord.SkyCoord(args.ra,args.dec,frame='icrs')

    stellar_model = simulacra.star.PhoenixModel(args.distance * u.pc,args.alpha,args.z,\
                                 args.temp,args.logg,target,\
                                 args.amplitude * u.km/u.s,args.period * u.day)

    return stellar_model

def get_tellurics(loc,wave_min,wave_max,args):
    tellurics_model = simulacra.tellurics.TelFitModel(wave_min,wave_max,loc)
    tellurics_model.pressure = args.pressure * u.Pa
    tellurics_model.temperature = args.temperature * u.Kelvin
    tellurics_model.humidity = args.humidity
    return tellurics_model


class CLI:
    """To add a new subcommand, just add a new classmethod and a docstring!"""
    _usage = None

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='A pipeline utility for running The Joker',
            usage=self._usage.strip())

        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])

        if not hasattr(self, args.command):
            print(f"Unsupported command '{args.command}'")
            parser.print_help()
            sys.exit(1)

        getattr(self, args.command)()


    def apogee(self):
        """Generate APOGEE data with a simple command."""
        parser = get_parser()
        parser = add_tellurics_args(parser)

        args = parser.parse_args(sys.argv[2:])
        obs = 'APO'
        loc = coord.EarthLocation.of_site(obs)
        # print(loc.lon,loc.latlat)
        stellar_model = get_star(loc,args)

        wave_min = 1.51*u.um
        wave_max = 1.70*u.um

        # Detector physical parameters
        ################################
        det_dict = {'resolution':22_500.0,
                    'area': np.pi * (2.5*u.m/2)**2,
                    'dark_current': 100/u.s,
                    'read_noise': 100,
                    'ccd_eff':0.99,
                    'through_put':0.05,
                    'epsilon': args.epsilon,
                    'w':args.w,
                    'gamma':args.gamma}

        tellurics_model = get_tellurics(loc,wave_min,wave_max,args)

        exp_times = np.ones(args.epoches)*args.exp_time * u.minute

        delta_x = simulacra.detector.spacing_from_res(1.4*det_dict['resolution'])
        x_grid = np.arange(np.log(wave_min.to(u.Angstrom).value),np.log(wave_max.to(u.Angstrom).value),delta_x)
        wave_grid = np.exp(x_grid) * u.Angstrom

        detector = simulacra.detector.Detector(stellar_model,loc=loc,wave_grid=wave_grid,**det_dict)
        data = run_simulation(detector,[tellurics_model],exp_times,args.epoches,args.window)

        filename = 'out/apogee_e{}_a{}_p{}'.format(args.epoches,stellar_model.amplitude.to(u.m/u.s).value,stellar_model.period.to(u.day).value)
        print(filename)
        data.to_h5(filename + '.h5')


    def keckhires(self):
        """Generate Keck HIRES data with a simple command."""
        # parser = argparse.ArgumentParser(sys.argv)

        parser = get_parser()
        parser = add_tellurics_args(parser)

        args = parser.parse_args(sys.argv[2:])
        obs = 'Keck Observatory'
        loc = coord.EarthLocation.of_site(obs)

        stellar_model = get_star(loc,args)

        wave_min = 500*u.nm
        wave_max = 630*u.nm

        # Detector physical parameters
        ################################
        det_dict = {'resolution':100_000.0,
                    'area': np.pi * (10*u.m/2)**2,
                    'dark_current': 100/u.s,
                    'read_noise': 100,
                    'ccd_eff':0.99,
                    'through_put':0.05,
                    'epsilon': args.epsilon,
                    'w':args.w,
                    'gamma':args.gamma}

        tellurics_model = get_tellurics(loc,wave_min,wave_max,args)

        gascell_model = simulacra.gascell.GasCellModel(filename='data/gascell/keck_fts_inUse.idl')

        exp_times = np.ones(args.epoches)*args.exp_time * u.minute

        delta_x = simulacra.detector.spacing_from_res(1.4*det_dict['resolution'])
        x_grid = np.arange(np.log(wave_min.to(u.Angstrom).value),np.log(wave_max.to(u.Angstrom).value),delta_x)
        wave_grid = np.exp(x_grid) * u.Angstrom

        detector = simulacra.detector.Detector(stellar_model,loc=loc,wave_grid=wave_grid,**det_dict)
        data = run_simulation(detector,[tellurics_model,gascell_model],exp_times,args.epoches,args.window)
        filename = 'out/keck_e{}_a{}_p{}'.format(args.epoches,stellar_model.amplitude.to(u.m/u.s).value,stellar_model.period.to(u.day).value)
        print(filename)
        data.to_h5(filename + '.h5')


    def expres(self):
        """Generate EXPRES data with a simple command."""
        parser = get_parser()
        parser = add_tellurics_args(parser)

        args = parser.parse_args(sys.argv[2:])
        obs = 'Lowell Observatory'
        loc = coord.EarthLocation.of_site(obs)

        stellar_model = get_star(loc,args)

        wave_min = 700*u.nm
        wave_max = 950*u.nm

        # Detector physical parameters
        ################################
        det_dict = {'resolution':130_000.0,
                    'area': np.pi * (4.3*u.m/2)**2,
                    'dark_current': 100/u.s,
                    'read_noise': 100,
                    'ccd_eff':0.99,
                    'through_put':0.05,
                    'epsilon': args.epsilon,
                    'w':args.w,
                    'gamma':args.gamma}

        tellurics_model = get_tellurics(loc,wave_min,wave_max,args)

        exp_times = np.ones(args.epoches)*args.exp_time * u.minute

        delta_x = simulacra.detector.spacing_from_res(1.4*det_dict['resolution'])
        x_grid = np.arange(np.log(wave_min.to(u.Angstrom).value),np.log(wave_max.to(u.Angstrom).value),delta_x)
        wave_grid = np.exp(x_grid) * u.Angstrom

        detector = simulacra.detector.Detector(stellar_model,loc=loc,wave_grid=wave_grid,**det_dict)
        data = run_simulation(detector,[tellurics_model],exp_times,args.epoches,args.window)

        filename = 'out/expres_e{}_a{}_p{}'.format(args.epoches,stellar_model.amplitude.to(u.m/u.s).value,stellar_model.period.to(u.day).value)
        print(filename)
        data.to_h5(filename + '.h5')


# Auto-generate the usage block:
cmds = []
maxlen = max([len(name) for name in CLI.__dict__.keys()])
for name, attr in CLI.__dict__.items():
    if not name.startswith('_'):
        cmds.append(f'    {name.ljust(maxlen)}  {attr.__doc__}\n')

CLI._usage = f"""
hq <command> [<args>]
Available commands:
{''.join(cmds)}
See more usage information about a given command by running:
    hq <command> --help
"""

# keck hires, gaia, apogee, expres

def main():
    CLI()
