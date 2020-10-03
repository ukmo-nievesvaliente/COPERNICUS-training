#!/usr/bin/env python

import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
#import seaborn as sns
from mpl_toolkits.axes_grid1 import AxesGrid
from collections import OrderedDict
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import wavetools.seastates.seastates as sst

##---
## helpers for plotting
def register_seastateCmaps(scale='douglas'):
    """ Register set of 'seastate' style colormaps. """

    if scale.lower() == 'douglas':
        # best equivalent to met office ops centre beaufort key
        Nval = 10
        cllist_byr = ['navy','blue','cyan','green','yellow',
                      'gold','red','darkred','black','white']
    if scale.lower() in ['beaufortsea','beaufort']:
        # from met office ops centre key
        Nval = 12
        cllist_byr = ['blue','dodgerblue','cyan','green','lime','yellow',
                      'gold','red','firebrick','darkred','black','white']

    cm = mpl_colors.LinearSegmentedColormap.from_list(
          scale.lower(), cllist_byr, N=Nval, gamma=1.0)       
    plt.cm.register_cmap(cmap=cm)

    return 

def setMapLatLons(extent):
    """Defines a lat-lon grid for a map"""

    lllon = extent[0]
    urlon = extent[1]
    lllat = extent[2]
    urlat = extent[3]

    llgridres = np.max([urlat-lllat,urlon-lllon])/5.0
    if llgridres < 1.0:
        llgridres = 0.5
    elif llgridres < 2.0:
        llgridres = np.ceil(llgridres)
    else:
        llgridres = np.ceil(llgridres/2.0) * 2.0

    # set meridians and parallels for map plot
    meridians = np.arange(lllon, urlon+llgridres, llgridres)
    parallels = np.arange(lllat, urlat+llgridres, llgridres)

    return meridians, parallels


def setMap(subplotpos, extent, meridians, parallels, resolution='50m', projection=ccrs.PlateCarree()):
    """Sets up a map for grid plotting"""

    map = plt.subplot(subplotpos, projection=projection)
    map.coastlines(resolution=resolution)
    map.set_xticks(meridians, crs=projection)
    map.set_yticks(parallels, crs=projection)
    lon_formatter = LongitudeFormatter(zero_direction_label=True, number_format='.1f')
    lat_formatter = LatitudeFormatter(number_format='.1f')
    map.xaxis.set_major_formatter(lon_formatter)
    map.yaxis.set_major_formatter(lat_formatter)
    map.gridlines(projection, draw_labels=False, xlocs=meridians, ylocs=parallels,linewidth=0.5)
    map.set_extent(extent, projection) 

    return map

##---
## plotting routines
def add_shippingareas(ax, usemap=False):

    shippingareas = sst.set_shippingareas()
    for key in shippingareas:
        locs = shippingareas[key]
        for lp in range(len(locs)):
            if lp == len(locs) - 1:
                # only plot area limits without 'False' field
                if usemap:
                    if len(locs[lp]) < 3:
                        ax.plot([locs[lp][1], locs[0][1]], [locs[lp][0], locs[0][0]], 
                                transform=ccrs.PlateCarree(), color='grey')
                else:
                    ax.plot([locs[lp][1], locs[0][1]], [locs[lp][0], locs[0][0]], color='grey')
            else:
                # only plot area limits without 'False' field
                if usemap:
                    if len(locs[lp]) < 3:
                        ax.plot([locs[lp][1], locs[lp+1][1]], [locs[lp][0], locs[lp+1][0]],
                                transform=ccrs.PlateCarree(), color='grey')
                else:
                    ax.plot([locs[lp][1], locs[lp+1][1]], [locs[lp][0], locs[lp+1][0]], color='grey')


def plot_pseastates(defseastates, pseastates, timename, cycstr=None, scale='douglas',
                    ptype='occur', show=False, save=False, savedir='.'):
    """Plots of seastate probabilities"""

    fig = plt.figure(figsize=(20,12), facecolor='white')
    cmap = 'coolwarm'
    #cmap = mpl_colors.ListedColormap(sns.color_palette("RdBu_r",11).as_hex())

    if scale.lower() == 'douglas':
        ncols = 5
        ssvar = 'sea-state'
        titleyloc = 0.96
    elif scale.lower() in ['beaufortsea','beaufort']:
        ncols = 6
        ssvar = 'wind force'
        titleyloc = 0.96
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(2, ncols),
                    axes_pad=0.30,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.20
                    )

    for lp, seastate in enumerate(defseastates):
        ax = grid[lp]
        #pfield = np.ma.masked_less(pseastates[lp,:,:], 0.01)
        pfield = pseastates[lp,:,:]
        clm = ax.pcolormesh(pfield,vmin=0.0,vmax=1.0,cmap=cmap)
        #ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(seastate)
    cbar = ax.cax.colorbar(clm)
    if ptype == 'encounter':
        supstr = 'Probability of encountering %s in window: %s' %(ssvar,timename)
    else:
        supstr = 'Probability of %s occurring in window: %s' %(ssvar,timename)
    if cycstr is not None:
        supstr = supstr + '\nIssued at %s' %cycstr
    plt.suptitle(supstr, y=titleyloc, fontsize=16)

    if save:
        fname = savedir + '/seastates_%s_%s_%s.png' %(scale, ptype, timename.replace(' ',''))
        plt.savefig(fname)
    if show:
        plt.show()
    plt.close()


def plot_mlseastates(*argv, cpseastate=None, timename='Undefined', cycstr=None, scale='douglas',
                     shipareas=None, usemap=False, show=False, save=False, savedir='.'):
    """Plots of most likely seastate/wind and probabilities
       Inputs:
        *argv [lats, lons] mlseastate - (optional) latitude, longitude and (most likely) sea-state array
        cpseastate - probabilities associated with most likely sea-state; generates second subplot
        timename - the time window used for the sea-state analysis
        scale - the sea-state scale used
        shipareas - apply defined shipping area to the map
        usemap - logical uses cartopy mapping fucntions"""

    if len(argv) == 3:
        lats = argv[0]
        lons = argv[1]
        mlseastate = argv[2]
    else:
        lats = None
        lons = None
        mlseastate = argv[0]

    plotprobs = False
    if cpseastate is not None: plotprobs = True

    defseastates = sst.set_seastates(scale=scale.lower())
    register_seastateCmaps(scale=scale.lower())

    if scale.lower() == 'douglas':
        sclmax = 9.5
    elif scale.lower() in ['beaufortsea','beaufort']:
        sclmax = 11.5
    
    if plotprobs:
        figxsize = 20
        ncols = 2
    else:
        figxsize = 12
        ncols = 1

    if (lats is not None) and (lons is not None):
        extent = [np.min(lons), np.max(lons), np.min(lats), np.max(lats)]
    if usemap:
        meridians, parallels = setMapLatLons(extent)

    fig = plt.figure(figsize=(figxsize,8), facecolor='white')

    subplotval = 100 + 10 * ncols + 1
    if usemap:
        ax = setMap(subplotval, extent, meridians, parallels, resolution='50m',
                      projection=ccrs.PlateCarree())
        ml = plt.pcolormesh(lons, lats, mlseastate-0.5, transform=ccrs.PlateCarree(), 
                            vmin=-0.5, vmax=sclmax, cmap=scale.lower())
        if shipareas is not None:
            add_shippingareas(ax, usemap=usemap)
    else:
        ax = fig.add_subplot(subplotval)
        if (lats is not None) and (lons is not None):
            ml = ax.pcolormesh(lons, lats, mlseastate-0.5, vmin=-0.5, vmax=sclmax, cmap=scale.lower())
            if shipareas is not None:
                add_shippingareas(ax, usemap=usemap)
            ax.set_xlim([extent[0],extent[1]])
            ax.set_ylim([extent[2],extent[3]])
        else:
            ml = ax.pcolormesh(mlseastate-0.5, vmin=-0.5, vmax=sclmax, cmap=scale.lower())
    cbar = plt.colorbar(ml, ticks=np.arange(len(defseastates)), shrink=0.6)
    cbar.ax.set_yticklabels(list(key for key in defseastates))
    if scale.lower() == 'beaufort':
        ax.set_title('Most likely Wind Category')
    else:
        ax.set_title('Most likely Sea-State')

    if plotprobs:
        #cmap = mpl_colors.ListedColormap(sns.color_palette("RdBu_r",11).as_hex())
        cmap = 'coolwarm'
        subplotval = 100 + 10 * ncols + 2
        if usemap:
            ax2 = setMap(subplotval, extent, meridians, parallels, resolution='50m',
                          projection=ccrs.PlateCarree())
            cp = ax2.pcolormesh(lons, lats, cpseastate, transform=ccrs.PlateCarree(),
                                 vmin=0, vmax=1.0, cmap=cmap)
            if shipareas is not None:
                add_shippingareas(ax2, usemap=usemap)
        else:
            ax2 = fig.add_subplot(subplotval)
            if (lats is not None) and (lons is not None):
                cp = ax2.pcolormesh(lons, lats, cpseastate, vmin=0, vmax=1.0, cmap=cmap)
                if shipareas is not None:
                    add_shippingareas(ax2, usemap=usemap)
                ax2.set_xlim([extent[0],extent[1]])
                ax2.set_ylim([extent[2],extent[3]])
            else:
                cp = ax2.pcolormesh(cpseastate, vmin=0, vmax=1.0, cmap=cmap)
        cbar = plt.colorbar(cp, shrink=0.6)
        ax2.set_title('Probability')

    titleyloc = 0.96
    if usemap: titleyloc = 0.86
    if scale.lower() == 'beaufort':
        supstr = 'Most likely wind category forecast: %s' %timename
    else:
        supstr = 'Most likely sea-state forecast: %s' %timename
    if cycstr is not None:
        supstr = supstr + '\nIssued at %s' %cycstr
    plt.suptitle(supstr, y=titleyloc, fontsize=16)

    if save:
        fname = savedir + '/seastates_%s_mostlikely_%s.png' %(scale, timename.replace(' ',''))
        plt.savefig(fname)
    if show:
        plt.show()
    plt.close()


def plot_pfield(*argv, pfield, timename='Undefined', cycstr=None, event=None, vmin=0.0,
                showzeros=False, shipareas=False, usemap=False, show=False, save=False, savedir='.'):
    """Plots of probabilities
       Inputs:
        *argv [lats, lons] pfield - (optional) latitude, longitude and probability field
        cpseastate - probabilities associated with most likely sea-state; generates second subplot
        timename - the time window used for the sea-state analysis
        scale - the sea-state scale used
        shipareas - apply defined shipping area to the map
        usemap - logical uses cartopy mapping fucntions"""

    if len(argv) == 2:
        lats = argv[0]
        lons = argv[1]
        #pfield = argv[2]
    else:
        lats = None
        lons = None
        #pfield = argv[0]

    if (lats is not None) and (lons is not None):
        extent = [np.min(lons), np.max(lons), np.min(lats), np.max(lats)]
    if usemap:
        meridians, parallels = setMapLatLons(extent)

    if vmin != 0.0:
        pfield.mask[pfield < vmin] = True
    elif not showzeros:
        # set slightly larger than zero to mask low finite probs
        pfield.mask[pfield <= 0.001] = True

    fig = plt.figure(figsize=(10,8), facecolor='white')
    #cmap = mpl_colors.ListedColormap(sns.color_palette("RdBu_r",11).as_hex())
    cmap = 'coolwarm'

    subplotval = 111
    if usemap:
        ax = setMap(subplotval, extent, meridians, parallels, resolution='50m',
                      projection=ccrs.PlateCarree())
        cp = ax.pcolormesh(lons, lats, pfield, transform=ccrs.PlateCarree(),
                             vmin=vmin, vmax=1.0, cmap=cmap)
        if shipareas:
            add_shippingareas(ax, usemap=usemap)
    else:
        ax = fig.add_subplot(subplotval)
        if (lats is not None) and (lons is not None):
            cp = ax.pcolormesh(lons, lats, pfield, vmin=vmin, vmax=1.0, cmap=cmap)
            if shipareas:
                add_shippingareas(ax, usemap=usemap)
            ax.set_xlim([extent[0],extent[1]])
            ax.set_ylim([extent[2],extent[3]])
        else:
            cp = ax.pcolormesh(pfield, vmin=vmin, vmax=1.0, cmap=cmap)
    cbar = plt.colorbar(cp, shrink=0.6)
    if event is not None:
        titlestr = '%s: Probability of %s' %(timename, event[1])
    else:
        titlestr = '%s: Probability' %timename
    if cycstr is not None:
        titlestr = titlestr + '\nIssued at %s' %cycstr
    ax.set_title(titlestr)

    if save:
        if event is not None:
            fname = savedir + '/seastates_%s_%s.png' %(event[0].replace(' ',''), timename.replace(' ',''))
        else:
            fname = savedir + '/seastates_pfield_%s.png' %timename.replace(' ','')
        plt.savefig(fname)
    if show:
        plt.show()
    plt.close()


def plot_field(field, timename, cycstr=None, fieldname=None, lats=None, lons=None, 
                 vmin=None, vmax=None, shipareas=False, show=False, save=False, savedir='.'):
    """Plots of fields"""

    if (lats is not None) and (lons is not None):
        usemap = True
        extent = [np.min(lons), np.max(lons), np.min(lats), np.max(lats)]
        meridians, parallels = setMapLatLons(extent)

    fig = plt.figure(figsize=(10,8), facecolor='white')

    subplotval = 111
    if usemap:
        ax = setMap(subplotval, extent, meridians, parallels, resolution='50m',
                      projection=ccrs.PlateCarree())
        cp = ax.pcolormesh(lons, lats, field, transform=ccrs.PlateCarree(),
                             vmin=vmin, vmax=vmax, cmap='viridis')
        if shipareas:
            add_shippingareas(ax)
    else:
        ax = fig.add_subplot(subplotval)
        cp = ax.pcolormesh(field, vmin=vmin, vmax=vmax, cmap='viridis')
    cbar = plt.colorbar(cp, shrink=0.6)
    if fieldname is not None:
        titlestr = '%s: %s' %(timename, fieldname)
    else:
        titlestr = '%s' %timename
    if cycstr is not None:
        titlestr = titlestr + '\nIssued at %s' %cycstr
    ax.set_title(titlestr)

    if save:
        fname = savedir + '/seastates_%s_%s.png' %(fieldname.replace(' ',''), timename.replace(' ',''))
        plt.savefig(fname)
    if show:
        plt.show()
    plt.close()
