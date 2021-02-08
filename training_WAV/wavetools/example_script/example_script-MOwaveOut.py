#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import standard libraries
import datetime
import numpy
import matplotlib.pyplot
import sys
import os

# setting-up the paths
root_dir = os.getcwd()
print(root_dir)

# insert path where the local libraries are located
sys.path.insert(0,root_dir)
sys.path.insert(0,'/net/home/h01/nvalient/nvalient-python/notebooks/COPERNICUS-training/training_WAV/')

# import local library
import wavetools.loaders.read_MOWaveOut as rdwv

"""
Created on Fri Feb  5 11:24:00 2021

 SCRIPT EXAMPLE TO LOAD AND CONCATENATE DETERMINISTIC MODEL OUTPUT

@author: nvalient
"""

dataDir = '/data/users/nvalient/T2013_14/RUN_RESULTS/UKC4aow'

# set the local data directory
datadir = os.path.join('dataDir')

# set out_dir to save the plots
outdir = os.path.join(root_dir,'img/')

# set model run name
model_run = 'UKC4aow'

# set year, month and day for the analysis cycle
# NOTE: at present the model cycles once per day, so hour is always set to zero (UTC)
# analysis/forecast data
year=2013
month=11
day=28
fcday=0

cycle=datetime.datetime(year,month,day,0)

# generate a NWS filename based on cycle time and which day we want (range[-1,5])
ncfile = rdwv.genfilename(model_run=model_run,cycle=cycle,fcday=fcday,datadir=dataDir)

# get the content of the chosen netCDF file
rdwv.contentWaveOut(ncfile)

# set a leadtime range (e.g., range [0,120] in hours, corresponding to 0 to 5 days)
# in days i.e., 3 days
leadtimes=[0,71]

# set the variable to retrieve (e.g. from list above)
varname = ['hs','t01','uust','vust']
VAR = []
#ia = 0
#var1 = rdwv.readWaveOut(varname[ia], model_run=model_run, cycle=cycle, leadtimes=leadtimes, datadir=dataDir)

# loop on variables
for ia in range(len(varname)):
    VAR.append(rdwv.readWaveOut(varname[ia], model_run=model_run, cycle=cycle, leadtimes=leadtimes, datadir=dataDir))
    # var is a class for the loaded data - show the available attributes
    print()
    print('var is a python object with the following keys:')
    print(VAR[ia].__dict__.keys())
    # print the shape of the loaded data [t,y,x]
    print('array shape for data loaded into var is as follows:')
    print(numpy.shape(VAR[ia].data))

# Quick plot to check it is working - hs
# set a time index
time_index = 25

# plot gridded field using pcolormesh
gvar = matplotlib.pyplot.pcolormesh(VAR[0].glons,VAR[0].glats,VAR[0].data[time_index,:,:],cmap='nipy_spectral')
cb = matplotlib.pyplot.colorbar()

# generate a title using the long name and validity time values
title = '%s: %s' %(VAR[0].longname, VAR[0].times[time_index].strftime('%Y-%m-%d %H:%MZ'))
matplotlib.pyplot.title(title)

# save the plot
out_name = os.path.join(outdir,'storm_Ex.png')
matplotlib.pyplot.savefig(out_name,bbox_inches="tight", pad_inches=0.1, dpi=150)

# show the plot
#matplotlib.pyplot.show()