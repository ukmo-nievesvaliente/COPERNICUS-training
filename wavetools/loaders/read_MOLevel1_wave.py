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
        self.fctype      = None  # deterministic or ensemble forecast type
        self.glats       = None  # latitude of model grid cell
        self.glons       = None  # longitude of model grid cell
        self.mbrs        = None  # numeric value of ensemble members - control is member 0
        self.mbr_names   = None  # string name for ensemble members
        self.times       = None  # datetime object for forecast validity times
        self.fcref       = None  # datetime object for forecast reference time (cycle time)
        self.fclead      = None  # numeric value for forecast lead time (in seconds)
        self.data        = None  # variable data
     
    def loadGrid(self, filein, var, fctype='deterministic', twindow=[0,None], xyindices=None):
        """Load data from Level1 deterministic or ensemble point product file"""
        if fctype.lower() not in ['deterministic', 'ensemble']:
            raise Exception('Please set fctype as either "deterministic" or "ensemble"')
        print('[INFO] Loading in data from %s' %filein)
        d = nc.Dataset(filein)
        self.var        = var
        self.longname   = d.variables[var].long_name
        self.fctype     = fctype.lower()
        self.glats      = d.variables['latitude'][:]
        self.glons      = d.variables['longitude'][:]
        self.times      = nc.num2date(d.variables['time'][twindow[0]:twindow[1]], 
                                       units=d.variables['time'].units)
        self.fclead     = d.variables['forecast_period'][twindow[0]:twindow[1]]
        if fctype.lower() == 'deterministic':
            self.data = d.variables[var][twindow[0]:twindow[1],:,:]
            # if supplied with a set of xy indices, collapse the array
            # that has been read in - this is much quicker than working from the file
            if xyindices is not None:
                self.data = self.data[:,xyindices[:,1],xyindices[:,0]]
        else:
            self.data = d.variables[var][:,twindow[0]:twindow[1],:,:]
            # if supplied with a set of xy indices, collapse the array
            # that has been read in - this is much quicker than working from the file
            if xyindices is not None:
                self.data = self.data[:,:,xyindices[:,1],xyindices[:,0]]
        if fctype.lower() == 'deterministic':
            self.fcref  = nc.num2date(
                          d.variables['forecast_reference_time'][:], 
                           units=d.variables['forecast_reference_time'].units)
        elif fctype.lower() == 'ensemble':
            self.mbrs      = d.variables['realization'][:]
            self.mbr_names = nc.chartostring(d.variables['ensemble_member_label'][:])
            self.fcref     = nc.num2date(
                              d.variables['realization_forecast_reference_time'][:], 
                              units=d.variables['realization_forecast_reference_time'].units)
        d.close()

    def concatenate_ens(self, var2):
        """Concatenate two variables along the realization (ensemble) axis"""
        print('[INFO] Concatenating ensemble members')
        self.data      = np.ma.concatenate((self.data,var2.data), axis=0)
        self.mbrs      = np.concatenate((self.mbrs,var2.mbrs), axis=0)
        self.mbr_names = np.concatenate((self.mbr_names,var2.mbr_names), axis=0)
        self.fcref     = np.concatenate((self.fcref,var2.fcref), axis=0)

    def concatenate_time(self, var2):
        """Concatenate two variables along the time axis"""
        print('[INFO] Concatenating time axis data')
        self.times      = np.concatenate((self.times,var2.times), axis=0)
        self.fclead     = np.concatenate((self.fclead,var2.fclead), axis=0)
        if self.fctype == 'deterministic':
            self.data = np.ma.concatenate((self.data,var2.data), axis=0)
        else:
            self.data = np.ma.concatenate((self.data,var2.data), axis=1)
 

# file naming and inspection

def genfilename(varname, cycle, fcday=0, ensemble=False, domain='uk', 
                datadir='./'):
    """Generates a Met Office Level1 file name"""

    cycstr = cycle.strftime('b%Y%m%dT%H00Z')
    fcdstr = (cycle + dt.timedelta(hours=fcday*24)).strftime('hi%Y%m%dT%H00Z')
    mtype = 'wave'
    if ensemble: mtype='waveen'
    fname = cycstr + '_' + fcdstr + '-' + mtype + '_' + domain.lower() + \
             '_standard_v1-level_1-' + varname + '.nc'
    ncfile = datadir + '/' + fname

    return ncfile


def contentWaveL1(ncfile):
    """Reads and prints contents of a wave level1 file"""

    print('[INFO] Reading content of file: %s' %ncfile)
    d = nc.Dataset(ncfile)
    print('title: %s' %d.title)
    print('source: %s' %d.source)
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
    t = nc.num2date(d.variables['time'][:], units=d.variables['time'].units)
    fct = d.variables['forecast_period'][:] / 3600
    for i,vt in enumerate(t):
        lstr = 'Leadtime (hours): %d' %fct[i]
        tstr = vt.strftime('VT: %Y-%m-%d %H:%M:%S')
        print('%s; %s' %(lstr, tstr))


# variable load functions

def setVarnames(collection='wave',vartype='level1'):
    """Defines variables names and associated file extensions"""

    varnames = OrderedDict()
    if collection.lower() == 'wave':
        if vartype.lower() == 'level1':
            varnames['VHM0'] = ['wave_significant_height']
            varnames['WSPD'] = ['wave_grid_10m_wind']
            varnames['WDIR'] = ['wave_grid_10m_wind']
            varnames['VTPK'] = ['wave_peak_period']
            varnames['VTM02'] = ['wave_period_tm02']
            varnames['VMDR'] = ['wave_mean_direction']
            varnames['VHM0_WW'] = ['wind_wave_significant_height']
            varnames['VTPK_WW'] = ['wind_wave_peak_period']
            varnames['VMDR_WW'] = ['wind_wave_mean_direction']
            varnames['VHM0_SW1'] = ['primary_swell_wave_significant_height']
            varnames['VTPK_SW1'] = ['primary_swell_wave_peak_period']
            varnames['VMDR_SW1'] = ['primary_swell_wave_mean_direction']
            varnames['VHM0_SW2'] = ['secondary_swell_wave_significant_height']
            varnames['VTPK_SW2'] = ['secondary_swell_wave_peak_period']
            varnames['VMDR_SW2'] = ['secondary_swell_wave_mean_direction']
            varnames['deptho'] = ['wave_grid_depth']

    return varnames


def readEnsWaveL1(varname, cycle, leadtimes=None, xyindices=None, lag=0, domain='uk', datadir='./'):
    """Read a variable from Level1 wave ensemble file"""

    varnames = setVarnames()

    cycle = cycle + dt.timedelta(hours=-1*lag)

    # construct list of times and file offsets based on leadtime inputs
    ntmax = 24
    tlist = []
    if leadtimes is not None:
        t0 = leadtimes[0] + lag
        t1 = leadtimes[1] + lag
        # the use of <= here means will always try to retrieve end leadtime
        while t0 <= t1:
            fcday = np.int(np.floor(t0 / ntmax))
            fc0 = np.mod(t0,ntmax)
            fc1 = np.min([t1-fcday*ntmax,ntmax])
            if fc1 == fc0: fc1 = fc0+1
            tlist.append([fcday, [fc0,fc1]])
            t0 = (fcday+1)*ntmax
    else:
        tlist.append([0, [0,None]])

    for i,tattr in enumerate(tlist):
        ncfile = genfilename(varnames[varname][0], cycle, fcday=tattr[0], 
                             ensemble=True, domain=domain, datadir=datadir)
        print('[INFO] Reading data from %s' %ncfile)
        if i == 0:
            try:
                var = varGrid()
                var.loadGrid(ncfile, varname, fctype='ensemble', twindow=tattr[1], xyindices=xyindices)
            except:
                print('[ERROR] Data not available for these lead times from %s' %ncfile)
        else:
            try:
                tmp = varGrid()
                tmp.loadGrid(ncfile, varname, fctype='ensemble', twindow=tattr[1], xyindices=xyindices)
                var.concatenate_time(tmp)
            except:
                print('[ERROR] Data not available for these lead times from %s' %ncfile)

    return var


def readWaveL1(varname, cycle, leadtimes=None, xyindices=None, domain='uk', datadir='./'):
    """Read a variable from Level1 deterministic wave file"""

    varnames = setVarnames()

    # construct list of times and file offsets based on leadtime inputs
    ntmax = 24
    tlist = []
    if leadtimes is not None:
        t0 = leadtimes[0]
        t1 = leadtimes[1]
        while t0 <= t1:
            fcday = np.int(np.floor(t0 / ntmax))
            fc0 = np.mod(t0,ntmax)
            fc1 = np.min([t1-fcday*ntmax,ntmax])
            if fc1 == fc0: fc1 = fc0+1
            tlist.append([fcday, [fc0,fc1]])
            t0 = (fcday+1)*ntmax
    else:
        tlist.append([0, [0,None]])

    for i,tattr in enumerate(tlist):
        ncfile = genfilename(varnames[varname][0], cycle, fcday=tattr[0], 
                             ensemble=False, domain=domain, datadir=datadir)
        print('[INFO] Reading data from %s' %ncfile)
        if i == 0:
            try:
                var = varGrid()
                var.loadGrid(ncfile, varname, fctype='deterministic', twindow=tattr[1], xyindices=xyindices)
            except:
                print('[ERROR] Data not available for these lead times from %s' %ncfile)
        else:
            try:
                tmp = varGrid()
                tmp.loadGrid(ncfile, varname, fctype='deterministic', twindow=tattr[1], xyindices=xyindices)
                var.concatenate_time(tmp)
            except:
                print('[ERROR] Data not available for these lead times from %s' %ncfile)

    return var


#def readCycles(var, cycle, ncycles=1, xyindices=None, datadir='./'):
#
#    print('[INFO] Reading wave data for variable %s' %var)
#    for icycle in range(ncycles):
#        leadtimes = [24*icycle,24*(icycle+1)]
#        if icycle == 0:
#            varGrid = wvrd.readWaveL1(var, cycle, leadtimes=None, xyindices=xyindices, domain='uk', datadir=datadir)
#        else:
#            tmpGrid = wvrd.readWaveL1(var, cycle, leadtimes=leadtimes, xyindices=xyindices, domain='uk', datadir=datadir)
#            varGrid.concatenate_time(tmpGrid)
#
#    return varGrid



