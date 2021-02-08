#!/usr/bin/env python

import netCDF4 as nc
import numpy as np
from collections import OrderedDict
import datetime as dt

##- classes and methods to read data

# variable grid class
class varGrid:

    def __init__(self):
        self.var         = None  # variable name
        self.longname    = None  # variable long name
        self.standard    = None  # variable CF standard name
        #self.fctype      = None  # deterministic or ensemble forecast type
        self.glats       = None  # latitude of model grid cell
        self.glons       = None  # longitude of model grid cell
        #self.mbrs        = None  # numeric value of ensemble members - control is member 0
        #self.mbr_names   = None  # string name for ensemble members
        self.times       = None  # datetime object for forecast validity times
        #self.fcref       = None  # datetime object for forecast reference time (cycle time)
        #self.fclead      = None  # numeric value for forecast lead time (in seconds)
        self.data        = None  # variable data

    def loadGrid(self, ncfile, var, twindow=[0,None], xyindices=None):
        """Load data from Wave model output with AMM15 grid"""
        print('[INFO] Loading in data from %s; %s' %(var,ncfile))
        d = nc.Dataset(ncfile)
        self.var        = var
        self.longname   = d.variables[var].globwave_name
        self.globname   = d.variables[var].globwave_name
        self.glats      = d.variables['latitude'][:]
        self.glons      = d.variables['longitude'][:]
        t = d.variables['time']
        t.set_always_mask(False)
        #tdt = nc.num2date(t[twindow[0]:twindow[1]], t.units)
        # model times can sometimes be out by a few microseconds
        # so round times to nearest minute:
        #tdt = np.vectorize(lambda tdt: (tdt + datetime.timedelta(seconds=1))
        #                   .replace(second=0, microsecond=0))(t)
        #self.times    = tdt
        self.times = nc.num2date(t[twindow[0]:twindow[1]], t.units)
        self.data = d.variables[var][twindow[0]:twindow[1],:,:]
        self.units    = d.variables[var].units
        # if supplied with a set of xy indices, collapse the array
        # that has been read in - this is much quicker than working from the file
        if xyindices is not None:
            self.data = self.data[:,xyindices[:,1],xyindices[:,0]]

        d.close()

    def concatenate_time(self, var2):
        """Concatenate two variables along the time axis"""
        print('[INFO] Concatenating time axis data')
        self.times      = np.concatenate((self.times,var2.times), axis=0)
        #self.fclead     = np.concatenate((self.fclead,var2.fclead), axis=0)
        self.data = np.ma.concatenate((self.data,var2.data), axis=0)

# file naming and inspection

def genfilename(model_run, cycle, fcday=0, domain='NWS',
                datadir='./'):
    """Generates a Met Office model output file name"""      

    cycstr = cycle.strftime('%Y%m%d00')
    #fcdstr = (cycle + dt.timedelta(hours=fcday*24)).strftime('hi%Y%m%dT%H00Z')
    fname = model_run + '_' + cycstr + '.nc'
    ncfile = datadir + '/' + fname

    return ncfile


def contentWaveOut(ncfile):
    d = nc.Dataset(ncfile)
    """Reads and prints contents of a wave model output (research) file"""

    print('[INFO] Reading content of file: %s' %ncfile)
    d = nc.Dataset(ncfile)
    print('---')
    print('Variables in file:')
    for i in d.variables:
        if 'long_name' in d.variables[i].ncattrs():
            print('%s : %s' %(i, d.variables[i].long_name))
    print('---')
    print('Validity times in file:')
    t = nc.num2date(d.variables['time'][:], units=d.variables['time'].units)
    print('%s' %(t))

# variable load functions

def setVarnamesOut(collection='wave',vartype='level1'):
    """Defines variables names and associated file extensions"""

    varnames = OrderedDict()
    if collection.lower() == 'wave':
        varnames['dpt'] = ['depth']
        varnames['hs'] = ['significant_wave_height']
        varnames['ucur'] = ['eastward_sea_water_velocity']
        varnames['vcur'] = ['northward_sea_water_velocity']
        varnames['uwnd'] = ['eastward_wind']
        varnames['vwnd'] = ['northward_wind']
        varnames['wlv'] = ['sea_surface_height_above_sea_level']
        varnames['t01'] = ['mean_period_t01']
        varnames['fp'] = ['dominant_wave_frequency']
        varnames['dir'] = ['wave_from_direction']
        varnames['wnm'] = ['mean_wave_number']
        varnames['uust'] = ['eastward_friction_velocity']
        varnames['vust'] = ['northward_friction_velocity']
        varnames['cha'] = ['charnock_coefficient']
        varnames['utaw'] = ['eastward_wave_supported_wind_stress']
        varnames['vtaw'] = ['northward_wave_supported_wind_stress']
        varnames['utwo'] = ['eastward_wave_to_ocean_stress']
        varnames['vtwo'] = ['northward_wave_to_ocean_stress']
        varnames['foc'] = ['wave_to_ocean_energy_flux']
        varnames['uuss'] = ['eastward_surface_stokes_drift']
        varnames['vuss'] = ['northward_surface_stokes_drift']
        varnames['sdc'] = ['drag_coefficient']
        varnames['twf'] = ['stress_fraction']
        varnames['fon'] = ['normalized_wave_to_ocean_energy_flux']
        varnames['utoc'] = ['eastward_ocean_stress']
        varnames['vtoc'] = ['northward_ocean_stress']

    return varnames

def readWaveOut(varname, model_run, cycle=None, leadtimes=None, xyindices=None, domain='NWS', datadir='./'):
    """Read a variable from wave model output (research) file"""

    #varnames = setVarnamesOut()
    
    tlist = []
    if leadtimes is not None:
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

    for i,tattr in enumerate(tlist):
        delta= dt.timedelta(days = 1)
        ncfile = genfilename(model_run, cycle, fcday=tattr[0], domain='NWS',
            datadir=datadir)

        if None not in tattr[1]:
            print('[INFO] Reading time indices %d to %d from %s' %(tattr[1][0],tattr[1][1],ncfile))
        else:
            print('[INFO] Reading all time indices from %s' %(ncfile))
        if i == 0:
            try:
                print('[Entering varGrid fun]')
                var = varGrid()
                var.loadGrid(ncfile, varname, twindow=tattr[1], xyindices=xyindices)
            except:
                print('[ERROR] Data not available for these lead times from %s' %ncfile)
        else:
            try:
                print('Entering varGrid fun - concatenate')
                tmp = varGrid()
                tmp.loadGrid(ncfile, varname, twindow=tattr[1], xyindices=xyindices)                
                var.concatenate_time(tmp)
            except:
                print('[ERROR] Data not available for these lead times from %s' %ncfile)
        # Loop over the number of days you want to concatenate
        cycle += delta

    return var    