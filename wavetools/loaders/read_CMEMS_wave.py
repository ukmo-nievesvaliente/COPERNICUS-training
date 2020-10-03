import netCDF4 as nc
import numpy as np
from collections import OrderedDict
import datetime as dt

##- classes and methods to read Copernicus Marine Environment Monitoring System wave model data

# set a class for gridded wave variables

class varGrid:

    def __init__(self):
        self.var      = None  # variable name
        self.longname = None  # variable long name
        self.standard = None  # variable CF standard name
        self.glats    = None  # latitude of model grid cell
        self.glons    = None  # longitude of model grid cell
        self.times    = None  # datetime object for forecast validity times
        self.fcref    = None  # datetime object for forecast reference time (cycle time)
        self.fclead   = None  # numeric value for forecast lead time (in seconds)
        self.data     = None  # variable data
        self.units    = None  # variable units
     
    def loadGrid(self, ncfile, var, twindow=[0,None], xyindices=None):
        """Load data from CMEMS wave product file"""
        print('[INFO] Loading %s data from %s' %(var,ncfile))
        d = nc.Dataset(ncfile)
        self.var      = var
        self.longname = d.variables[var].long_name
        self.standard = d.variables[var].standard_name
        self.glats    = d.variables['latitude'][:]
        self.glons    = d.variables['longitude'][:]
        # time data mask needs to be set to false to avoid array bug
        # this also rounds float data correctly to nearest minute
        t = d.variables['time']
        t.set_always_mask(False)
        tdt = nc.num2date(t[twindow[0]:twindow[1]], t.units)
        # model times can sometimes be out by a few microseconds
        # so round times to nearest minute:
        #tdt = np.vectorize(lambda tdt: (tdt + datetime.timedelta(seconds=1))
        #                   .replace(second=0, microsecond=0))(t)
        #self.times    = tdt
        self.times = nc.num2date(t[twindow[0]:twindow[1]], t.units)
        if 'forecast_period' in d.variables:
            self.fclead   = d.variables['forecast_period'][twindow[0]:twindow[1]]
        self.data     = d.variables[var][twindow[0]:twindow[1],:,:]
        self.units    = d.variables[var].units
        # if supplied with a set of xy indices, collapse the array
        # that has been read in - this is much quicker than working from the file
        if xyindices is not None:
            self.data = self.data[:,xyindices[:,1],xyindices[:,0]]
        if 'forecast_reference_time' in d.variables:
            self.fcref = nc.num2date(d.variables['forecast_reference_time'][:], 
                           units=d.variables['forecast_reference_time'].units)
        d.close()

    def concatenate_time(self, var2):
        """Concatenate two variables along the time axis"""
        print('[INFO] Concatenating time axis data')
        self.times = np.concatenate((self.times,var2.times), axis=0)
        if self.fclead is not None:
            self.fclead = np.concatenate((self.fclead,var2.fclead), axis=0)
        self.data = np.ma.concatenate((self.data,var2.data), axis=0)


# file naming and inspection

def genfilename(cycle, fcday=0, mfc='metoffice', cfg='amm15', domain='NWS', 
                reanalysis=False, datadir='.'):
    """Generates a CMEMS style file name based on input cycle and forecast day"""

    cycstr = cycle.strftime('b%Y%m%d')
    fcdstr = (cycle + dt.timedelta(hours=fcday*24)).strftime('hi%Y%m%d')
    if reanalysis:
        fname = mfc + '_wave_' + cfg + '_' + domain + '_WAV_3' + \
                fcdstr + '.nc'
    else:
        fname = mfc + '_wave_' + cfg + '_' + domain + '_WAV_' + \
                cycstr + '_' + fcdstr + '.nc'
    ncfile = datadir + '/' + fname

    return ncfile


def contentWaveCMEMS(ncfile):
    """Reads and prints contents of a CMEMS wave file"""

    print('[INFO] Reading content of file: %s' %ncfile)
    d = nc.Dataset(ncfile)
    print('title: %s' %d.title)
    print('source: %s' %d.source)
    if 'forecast_reference_time' in d.variables:
        t0 = nc.num2date(d.variables['forecast_reference_time'][0],
                         units=d.variables['forecast_reference_time'].units)
        tstr = t0.strftime('%Y-%m-%d %H:%M:%S')
        print('analysis/cycle time: %s' %tstr)
    print('---')
    print('Variables in file:')
    for i in d.variables:
        if 'long_name' in d.variables[i].ncattrs():
            print('%s : %s' %(i, d.variables[i].long_name))
    print('---')
    print('Validity times in file:')
    # time data mask needs to be set to false to avoid array bug
    # this also rounds float data correctly to nearest minute
    t = d.variables['time']
    t.set_always_mask(False)
    tdt = nc.num2date(t[:], t.units)
    for i,vt in enumerate(tdt):
        tstr = vt.strftime('VT: %Y-%m-%d %H:%M:%S')
        if 'forecast_period' in d.variables:
            fct = d.variables['forecast_period'][i] / 3600
            lstr = 'Leadtime (hours): %d' %fct
            tstr = '%s; %s' %(lstr, tstr)
        print('%s' %(tstr))
    d.close()


# variable loading

def readWaveCMEMS(varname, cycle=None, filename=None, reanalysis=False, leadtimes=None, xyindices=None,
                  mfc='metoffice', cfg='amm15', domain='NWS', datadir='.'):
    """Read a variable from CMEMS wave file"""

    # construct list of times and file offsets based on leadtime inputs
    tlist = []
    if filename is None:
        if not reanalysis and leadtimes is not None:
            ntmax = 24
            t0 = leadtimes[0]
            t1 = leadtimes[1]
            # the use of <= here means will always try to retrieve end leadtime
            while t0 <= t1:
                fcday = np.int(np.floor(t0 / ntmax))
                fc0 = np.mod(t0,ntmax)
                fc1 = np.min([t1-fcday*ntmax+1,ntmax])
                # case where both lead times are the same
                if fc1 == fc0: fc1 = fc0+1
                # case where second lead time is last index in array
                if fc1 == 0: fc1 = None
                tlist.append([fcday, [fc0,fc1]])
                t0 = (fcday+1)*ntmax
        else:
            tlist.append([0, [0,None]])
    else:
        tlist.append([0, [0,None]])

    for i,tattr in enumerate(tlist):
        if filename is None:
            ncfile = genfilename(cycle, fcday=tattr[0], mfc=mfc, cfg=cfg, domain=domain, 
                                 reanalysis=reanalysis, datadir=datadir)
        else:
            ncfile = datadir + '/' + filename
        if None not in tattr[1]:
            print('[INFO] Reading time indices %d to %d from %s' %(tattr[1][0],tattr[1][1],ncfile))
        else:
            print('[INFO] Reading all time indices from %s' %(ncfile))
        if i == 0:
            try:
                var = varGrid()
                var.loadGrid(ncfile, varname, twindow=tattr[1], xyindices=xyindices)
            except:
                print('[ERROR] Data not available for these lead times from %s' %ncfile)
        else:
            try:
                tmp = varGrid()
                tmp.loadGrid(ncfile, varname, twindow=tattr[1], xyindices=xyindices)
                var.concatenate_time(tmp)
            except:
                print('[ERROR] Data not available for these lead times from %s' %ncfile)

    return var
