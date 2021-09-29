from simulacra.theory import TheoryModel
import numpy.random as random
import astropy.units as u
import telfit
import numpy as np

def get_skycalc_defaults(inputFilename,almFilename,isVerbose=False):
    dic = {}

    # Query the Almanac if alm option is enabled
    if almFilename:

        # Read the input parameters
        inputalmdic = None
        try:
            with open(almFilename, 'r') as f:
                inputalmdic = json.load(f)
        except ValueError:
            with open(almFilename, 'r') as f:
                inputalmdic = sky_cli.loadTxt(f)

        if not inputalmdic:
            raise ValueError('Error: cannot read' + almFilename)

        alm = skycalc.AlmanacQuery(inputalmdic)
        dic = alm.query()

    if isVerbose:
        print('Data retrieved from the Almanac:')
        for key, value in dic.items():
            print('\t' + str(key) + ': ' + str(value))

    if inputFilename:

        # Read input parameters
        inputdic = None
        try:
            with open(inputFilename, 'r') as f:
                inputdic = json.load(f)
        except ValueError:
            with open(inputFilename, 'r') as f:
                inputdic = sky_cli.loadTxt(f)

        if not inputdic:
            raise ValueError('Error: cannot read ' + inputFilename)

        # Override input parameters
        if isVerbose:
            print('Data overridden by the user\'s input file:')
        for key, value in inputdic.items():
            if isVerbose and key in dic:
                print('\t' + str(key) + ': ' + str(value))
            dic[key] = value

    # Fix the observatory to fit the backend
    try:
        dic = sky_cli.fixObservatory(dic)
    except ValueError:
        raise

    if isVerbose:
        print('Data submitted to SkyCalc:')
        for key, value in dic.items():
            print('\t' + str(key) + ': ' + str(value))

    return dic

def sample_airmass(epoches):
    airmass = (2 * random.rand(epoches)) + 1
    return airmass


class TelluricsModel(TheoryModel):
    def __init__(self):
        pass
        # self._airmass = np.array([])

    # def generate_airmass(self,epoches,airmassmin=1.0,airmassmax=3.0):
    #     self._airmass = sample_airmass(epoches)

    # @property
    # def airmass(self,airmass):
        return self._airmass


class SkyCalcModel(TelluricsModel):
    def __init__(inputfile,almfile):
        super(SkyCalcModel,self).__init__()
        self.inputfile = inputfile
        self.almfile   = almfile

    def generate_transmission(epoches):
        # airmass = sample_airmass(epoches)
        dic     = get_skycalc_defaults(self.inputfile,self.almfile)
        lambda_grid       = []
        transmission_grid = []
        for a in self.airmass:
            dic['airmass'] = a
            skyModel = skycalc.SkyModel()
            skyModel.callwith(dic)

            data = skyModel.getdata()
            data = io.BytesIO(data)
            hdu = at.Table.read(data)
            transmission_grid.append(hdu['trans'].data)
            lambda_grid.append(hdu['lam'].data)
        self.flux = np.array(transmission_grid)
        self.wave = np.array(lambda_grid) * u.nm
        return np.array(transmission_grid), np.array(lambda_grid) * u.nm

def airmass_to_angle(airmass):
    return np.arccos(1./airmass) * 180/np.pi

class TelFitModel(TelluricsModel):
    def __init__(self,lambmin,lambmax,humidity_guess=0.4):
        self._lambmin = lambmin
        self._lambmax = lambmax
        self.humidity_guess = humidity_guess
        self.color = 'blue'

    def generate_transmission(self,epoches):
        self.airmass = sample_airmass(epoches)

        wavestart = self.lambmin/u.nm - 2.
        waveend = self.lambmax/u.nm + 2.
        modeler = telfit.Modeler()

        # print(wavestart, waveend)

        flux = np.array([])
        wave = np.array([])
        for a in self.airmass:
            angle = airmass_to_angle(a)
            model = modeler.MakeModel(humidity=self.humidity_guess,
                         lowfreq=1e7/waveend,
                         highfreq=1e7/wavestart,
                         angle=angle)
            ns = len(model.x)
            flux = np.concatenate((flux,model.y))
            wave = np.concatenate((wave,model.x * u.nm))

        self.flux = flux.reshape(epoches,ns)
        self.wave = wave.reshape(epoches,ns)
        # print('tel flux',self.flux.shape)
        # print('tel wave',self.wave.shape)
        return self.flux, self.wave

    @TheoryModel.lambmin.setter
    def lambmin(self,lambmin):
        self._lambmin = lambmin

    @TheoryModel.lambmax.setter
    def lambmax(self,lambmax):
        self._lambmax = lambmax
