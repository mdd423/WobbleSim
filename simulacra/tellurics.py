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

class TelFitModel(TelluricsModel):
    def __init__(self,lambmin,lambmax,airmass,loc,humidity=50.0,temperature=300*u.Kelvin,pressure=1.0e6*u.Pa,ns=100000):
        self.lambmin = lambmin
        self.lambmax = lambmax
        # assert that all of these parameters that are arrays
        # are the same length
        # or they are just a scalar
        self.epoches     = airmass.shape[0]
        self.airmass     = airmass
        self.humidity    = humidity
        self.temperature = temperature
        self.pressure    = pressure

        # dlamb = 1e-2 * u.Angstrom
        # self.wave = np.arange(lambmin.to(u.Angstrom).value,lambmax.to(u.Angstrom).value,step=dlamb.to(u.Angstrom).value) * u.Angstrom

        self.loc = loc

        self.color = 'blue'
        self.parameters = {}

    def temperature():
        doc = "The temperature property."
        def fget(self):
            return self._temperature
        def fset(self, value):
            try:
                temp = iter(value)
                self._temperature = value
            except TypeError:
                self._temperature = value * np.ones(self.epoches)
        def fdel(self):
            del self._temperature
        return locals()
    temperature = property(**temperature())

    def humidity():
        doc = "The humidity property."
        def fget(self):
            return self._humidity
        def fset(self, value):
            try:
                temp = iter(value)
                self._humidity = value
            except TypeError:
                self._humidity = value * np.ones(self.epoches)
        def fdel(self):
            del self._humidity
        return locals()
    humidity = property(**humidity())

    def pressure():
        doc = "The pressure property."
        def fget(self):
            return self._pressure
        def fset(self,value):
            try:
                temp = iter(value)
                self._pressure = value
            except TypeError:
                self._pressure = value * np.ones(self.epoches)
        def fdel(self):
            del self._pressure
        return locals()
    pressure = property(**pressure())

    def airmass():
        doc = "The airmass property."
        def fget(self):
            return self._airmass
        def fset(self, value):
            try:
                temp = iter(value)
                self._airmass = value
            except TypeError:
                self._airmass = value * np.ones(self.epoches)
            self._airmass = value * np.ones(self.epoches)
        def fdel(self):
            del self._airmass
        return locals()
    airmass = property(**airmass())

    def epoches():
        doc = "The epoches property."
        def fget(self):
            return self._epoches
        def fset(self,value):
            self._epoches = value
        def fdel(self):
            del self._epoches
        return locals()
    epoches = property(**epoches())

    def generate_transmission(self,times):

        modeler = telfit.Modeler(debug=True)
        flux = np.array([])
        wave = np.array([])
        print(self.lambmin.to(u.cm),self.lambmax.to(u.cm))
        for i,time in enumerate(times):

            angle = np.arccos(1./self.airmass[i]) * 180 * u.deg/np.pi
            print('humidity: {}\n'.format(self.humidity[i]),
                'pressure: {}\n'.format(self.pressure[i].to(u.hPa).value),
                'temperature: {}\n'.format(self.temperature[i].to(u.Kelvin).value),
                'lat: {}\n'.format(self.loc.lat.to(u.degree).value),
                'elevation: {}\n'.format(self.loc.height.to(u.km).value),
                'freqmin(cm-1): {}\n'.format(1.0/(self.lambmax.to(u.cm).value)),
                'freqmax(cm-1): {}\n'.format(1.0/(self.lambmin.to(u.cm).value)),
                'angle: {}\n'.format(angle.to(u.deg).value))

            model = modeler.MakeModel(humidity=self.humidity[i],
                         pressure=self.pressure[i].to(u.hPa).value,
                         temperature=self.temperature[i].to(u.Kelvin).value,
                         lat=self.loc.lat.to(u.degree).value,
                         alt=self.loc.height.to(u.km).value,
                         lowfreq=(1.0/self.lambmax.to(u.cm).value),
                         highfreq=(1.0/self.lambmin.to(u.cm).value),
                         angle=angle.to(u.deg).value)

            ns   = len(model.x)
            print(ns)
            flux = np.concatenate((flux,model.y))
            wave = np.concatenate((wave,model.x * u.nm))

        flux = flux.reshape(self.epoches,ns)
        wave = wave.reshape(self.epoches,ns)
        return flux, wave

    @TheoryModel.lambmin.setter
    def lambmin(self,lambmin):
        self._lambmin = lambmin

    @TheoryModel.lambmax.setter
    def lambmax(self,lambmax):
        self._lambmax = lambmax
