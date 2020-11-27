#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 16:52:45 2020

@author: nvalient
"""
# import standard libraries
import datetime
import numpy
import matplotlib.pyplot
import os
import sys
import math
#import cartopy.crs as ccrs
#from matplotlib.animation import FuncAnimation

# setting-up the paths
root_dir = os.getcwd()
print(root_dir)

# insert path where the local library is located
sys.path.insert(0,root_dir)

# import local library
import wavetools.loaders.read_CMEMS_wave as rdwv

# set the local data directory
datadir = os.path.join(root_dir,'data')
out_dir = '/net/home/h01/nvalient/nvalient-python/COPERNICUS-training/plots_ppt/slide_plots'

# set year, month and day for the analysis cycle
# NOTE: at present the model cycles once per day, so hour is always set to zero (UTC)
# storm Aiden - analysis/forecast data
year=2020
month=10
day=28
fcday=0

#storm Dennis - analysis data
#year=2020
#month=2
#day=16
# downloaded file is the analysis from 16/02/2020, i.e. fields from the 15th
# so forecast day is set to -1
#fcday=-1

cycle=datetime.datetime(year,month,day,0)

# generate a NWS filename based on cycle time and which analysis/forecast day we want (range[-1,5])
ncfile = rdwv.genfilename(cycle,fcday=fcday,datadir=datadir)

# get the content of the chosen netCDF file
rdwv.contentWaveCMEMS(ncfile)

# set a leadtime range (possible range [-24,120] in hours, corresponding to -1 to 5 days for 20200928 dataset); for tha case of Aiden (20201028),
#  we have only from [0,47]
leadtimes=[0,47]

# if using the storm Dennis example range is [-24,-1]
#leadtimes = [-24,-1]

# set the variable to retrieve (e.g. from list above)

var1 = rdwv.readWaveCMEMS('VHM0', cycle=cycle, leadtimes=leadtimes, datadir=datadir)
var2 = rdwv.readWaveCMEMS('VMDR', cycle=cycle, leadtimes=leadtimes, datadir=datadir)

#wspd = np.sqrt(uwnd**2.0 + vwnd**2.0)
#wdir = np.arctan2(vwnd,uwnd)amm15_20200227_t11.png

# to use in quiver
Mdir = var2.data[:,:,:]
def MdirCopernicus2zonal(Mdir):  
    """ Copernicus wave direction is Mean wave direction FROM (Mdir) wrt the N (we call it phi)
    
    In order to get the x,y components (for quiver) we need theta (wrt the zonal direction; i.e., E = 0 deg angle).
    Direction angles gamma are wrt True North: the angle wrt the zonal direction theta is 
                                               theta = 90.-gamma
    Direction angles theta gives where waves are going: the angles where waves are coming from phi will be gamma+180; therefore 
    gamma = phi - 180;
    Combining the two we have that the angle theta we want is
    theta = 90 - (phi - 180) """
    theta = 270. - Mdir
    return theta

def getXY_MdirCopernicus(theta):
    """ To use in quiver - Get meridional and zonal components from Mdir (WAV Copernicus products)
        Inputs: Theta, Mdir corrected wrt zonal direction + direction To (not from)
        Outputs: x_hs; zonal component for Hs direction
                 y_hs; meridional component for Hs direction """
    x_hs = 1. * numpy.cos((theta)*math.pi/180.)
    y_hs = 1. * numpy.sin((theta)*math.pi/180.)
    
    return x_hs, y_hs

# var is a class for the loaded data - show the available attributes
print()
print('var is a python object with the following keys:')
print(var1.__dict__.keys())
# print the shape of the loaded data [t,y,x]
print('array shape for data loaded into var is as follows:')
print(numpy.shape(var1.data))

# plot gridded field using pcolormesh
cmap = 'nipy_spectral'

theta = MdirCopernicus2zonal(Mdir)
x_hs, y_hs = getXY_MdirCopernicus(theta)

# Start loop for plots
for it in range(numpy.shape(var1.data)[0]):
    fig = matplotlib.pyplot.figure(figsize=(5, 5),facecolor='white')
    axes = fig.add_subplot(111)#,projection=ccrs.RotatedPole(pole_latitude=37.5, pole_longitude=177.5))
    #fig, ax = matplotlib.pyplot.subplots([111],projection=ccrs.RotatedPole(pole_latitude=37.5, pole_longitude=177.5))
    gvar  = axes.pcolormesh(var1.glons,var1.glats,var1.data[it,:,:],cmap = cmap, vmin = 0, vmax = 14)
    cb = matplotlib.pyplot.colorbar(gvar,orientation='horizontal') # put colorbar horizontal
    cb.set_label('Hs [m]',size=12)
    gvar2 = matplotlib.pyplot.quiver(var2.glons[::49],var2.glats[::49],
                                     x_hs[it,::49,::49],y_hs[it,::49,::49],
                                     units='xy',
                                     angles='xy',
                                     scale = 7.,
                                     scale_units = 'inches')
    #axes.coastlines(resolution='50m', color='black', linewidth=1)
    #axes.add_feature(land_50)
    # generate a title using the long name and validity time values
    title = '%s: %s' %(var1.longname, var1.times[it].strftime('%Y-%m-%d %H:%MZ'))
    matplotlib.pyplot.title(title)
    #matplotlib.pyplot.show()
    
    # Save plots
    # '%02d' % it
    #anim = FuncAnimation(fig,animate,frames=niter,blit=True,interval=150)
    #anim.save('AidenStorm.gif', dpi=72, writer='imagemagick')
    
    out_name = os.path.join(out_dir,"{0:0=2d}".format(it)+'-Aiden.png')
    print('Saving '+"{0:0=2d}".format(it)+'-Aiden.png')
    matplotlib.pyplot.savefig(out_name,bbox_inches="tight", pad_inches=0.1, dpi=150)
    matplotlib.pyplot.close()