#!/usr/bin/env python

import numpy as np
from collections import OrderedDict
import datetime as dt

##---
## dictionaries
def set_leadtimes():
    """Create an ordered dictionary of forecast time terms for Shipping forecasts"""

    defleadtimes = OrderedDict()
    defleadtimes['Imminent'] = [0,6]
    defleadtimes['Soon'] = [6,12]
    defleadtimes['Later'] = [12,24]
    defleadtimes['Tomorrow'] = [24,48]
    defleadtimes['Day 2 Outlook'] = [48,72]
    defleadtimes['Day 3 Outlook'] = [72,96]
    defleadtimes['Day 4 Outlook'] = [96,120]
    defleadtimes['Day 5 Outlook'] = [120,144]
    defleadtimes['Today'] = [0,23] # for analysing 00-23Z on a given day
    defleadtimes['Yesterday'] = [-24,-1] # for analysing 00-23Z on previous day, e.g. CMEMS analysis data

    return defleadtimes


def set_seastates(scale='douglas'):
    """Create ordered dictionary of seastate name and thresholds"""

    print('[INFO] Setting analysis scale to: %s' %scale)
    defseastates = OrderedDict()

    if scale.lower() == 'douglas':
        defseastates['Calm (glassy)']=[0.00,0.01]
        defseastates['Calm (rippled)']=[0.01,0.10]
        defseastates['Smooth']=[0.10,0.50]
        defseastates['Slight']=[0.50,1.25]
        defseastates['Moderate']=[1.25,2.50]
        defseastates['Rough']=[2.50,4.00]
        defseastates['Very Rough']=[4.00,6.00]
        defseastates['High']=[6.00,9.00]
        defseastates['Very High']=[9.00,14.00]
        defseastates['Phenomenal']=[14.00,50.0]
    if scale.lower() == 'beaufortsea':
        defseastates['Calm (glassy)']=[0.00,0.06] #centre <0.1
        defseastates['Calm (rippled)']=[0.05,0.15] #centre 0.1
        defseastates['Smooth']=[0.15,0.35] #centre 0.2
        defseastates['Slight']=[0.35,0.8] # centre 0.6
        defseastates['Slight-Moderate']=[0.8,1.5] #centre 1.0
        defseastates['Moderate']=[1.5,2.5] #centre 2.0
        defseastates['Rough']=[2.5,3.5] #centre 3.0
        defseastates['Rough-Very Rough']=[3.5,4.75] #centre 4.0
        defseastates['Very Rough-High']=[4.75,6.5] #centre 5.5
        defseastates['High']=[6.5,9.0] # centre 7.0
        defseastates['Very High']=[9.0,13.5] # centre 10.0
        defseastates['Phenomenal']=[13.5,50.0] # centre 14.0
    if scale.lower() == 'beaufort':
        defseastates['Calm']=[0.00,0.55]
        defseastates['Light Air']=[0.55,1.55]
        defseastates['Light Breeze']=[1.55,3.35]
        defseastates['Gentle Breeze']=[3.35,5.55]
        defseastates['Moderate Breeze']=[5.55,7.95]
        defseastates['Fresh Breeze']=[7.95,10.75]
        defseastates['Strong Breeze']=[10.75,13.85]
        defseastates['Near Gale']=[13.85,17.15]
        defseastates['Gale']=[17.15,20.75]
        defseastates['Severe Gale']=[20.75,24.45]
        defseastates['Storm-Violent Storm']=[24.45,32.65]
        defseastates['Hurricane']=[32.65,200.0]

    return defseastates


def set_shippingareas():
    """Create ordered dictionary of shipping and high seas areas"""

    shippingareas = OrderedDict()
    # Met Office shipping areas
    shippingareas['Viking'] = [[58.5,0.0],[61.0,0.0],[61.0,4.0],[58.5,4.0]]
    shippingareas['North Utsire'] = [[59.0,4.0],[61.0,4.0],[61.0,5.0,False],[59.0,5.6]]
    shippingareas['South Utsire'] = [[57.75,4.0],[59.0,4.0],[59.0,5.6,False],[58.0,7.083],[57.75,7.5]]
    shippingareas['Forties'] = [[56.0,-1.0],[58.5,-1.0],[58.5,4.0],[56.0,4.0]]
    shippingareas['Forth'] = [[55.66,-1.5,False],[57.0,-2.166],[57.0,-1.0],[56.0,-1.0]] #add nodes
    shippingareas['Cromarty'] = [[57.0,-2.166,False],[58.5,-3.0],[58.5,-1.0],[57.0,-1.0]] #add nodes
    shippingareas['Tyne'] = [[54.25,-0.33,False],[55.66,-1.5],[56.0,-1.0],[54.25,0.75]] #add nodes
    shippingareas['Dogger'] = [[54.25,0.75],[56.0,-1.0],[56.0,4.0],[54.25,4.0]]
    shippingareas['Fisher'] = [[56.0,4.0],[57.75,4.0],[57.75,7.5],[57.08,8.59,False],[56.0,8.16]]
    shippingareas['German Bight'] = [[54.25,4.0],[56.0,4.0],[56.0,8.16,False],[53.89,8.97,False],[52.75,4.66],[53.59,4.66]] #add nodes
    shippingareas['Humber'] = [[52.75,1.66,False],[54.25,-0.33],[54.25,4.0],[53.59,4.66],[52.75,4.66]] #add nodes
    shippingareas['Thames'] = [[51.25,1.42,False],[52.75,1.66],[52.75,4.66,False],[51.25,2.92]] #add nodes
    shippingareas['Dover'] = [[50.75,0.25,False],[51.25,1.42],[51.25,2.92,False],[50.25,1.5]] #add nodes
    shippingareas['Wight'] = [[49.75,-1.92],[50.59,-1.92,False],[50.75,0.25],[50.25,1.5,False]] #add nodes
    shippingareas['Portland'] = [[48.83,-3.5],[50.75,-3.5,False],[50.59,-1.92],[49.75,-1.92,False]] #add nodes
    shippingareas['Plymouth'] = [[48.45,-6.25],[50.0,-6.25],[50.09,-5.75,False],[50.42,-3.5],[48.83,-3.5,False],[48.45,-4.75]] #check nodes
    shippingareas['Biscay'] = [[43.59,-6.25],[48.45,-6.25],[48.45,-4.75,False]] #add nodes
    shippingareas['Trafalgar'] = [[35.0,-15.0],[41.0,-15.0],[41.0,-8.66],[35.0,-6.25]]
    shippingareas['Fitzroy'] = [[41.0,-15.0],[48.45,-15.0],[48.45,-6.25],[43.59,-6.25],[41.0,-8.66]]
    shippingareas['Sole'] = [[48.45,-15.0],[50.0,-15.0],[50.0,-6.25],[48.45,-6.25]]
    shippingareas['Lundy'] = [[50.0,-6.25],[52.5,-6.25],[52.0,-5.09,False],[50.09,-5.75]] #add nodes
    shippingareas['Fastnet'] = [[50.0,-10.0],[51.59,-10.0,False],[52.5,-6.25],[50.0,-6.25]]
    shippingareas['Irish Sea'] = [[52.5,-6.25,False],[54.75,-5.75],[54.83,-5.09,False],[52.0,-5.09]] #add nodes
    shippingareas['Bailey'] = [[58.0,-15.0],[62.42,-15.0],[60.59,-10.0],[58.0,-10.0]]
    shippingareas['Shannon'] = [[50.0,-15.0],[53.5,-15.0],[53.5,-10.09,False],[51.59,-10.0],[50.0,-10.0]] #add nodes
    shippingareas['Rockall'] = [[53.5,-15.0],[58.0,-15.0],[58.0,-10.0],[54.33,-10.0,False],[53.5,-10.09]]
    shippingareas['Malin'] = [[54.33,-10.0],[57.0,-10.0],[57.0,-5.83,False],[54.83,-5.09],[54.75,-5.75,False]] #add nodes
    shippingareas['Hebrides'] = [[57.0,-10.0],[60.59,-10.0],[58.66,-5.0,False],[57.0,-5.83]]
    shippingareas['Faeroes'] = [[61.16,-11.5],[63.33,-7.5],[61.83,-2.5],[59.5,-7.25]]
    shippingareas['Fair Isle'] = [[58.5,-3.0],[58.66,-5.0],[59.5,-7.25],[61.83,-2.5],[61.0,0.0],[58.5,0.0]]
    shippingareas['Southeast Iceland'] = [[61.16,-11.5],[63.59,-18.0],[65.0,-13.59],[63.33,-7.5]]

    ##high seas
    #DENMARK STRAIT
    #  71 00 N 21 40'W  71 00'N 20 00'W  66 00'N 20 00'W  65 00'N 22 50'W  65 00'N 40 00'W	 
    #NORTH ICELAND
    #  71 00'N 20 00'W  71 00'N 10 00'W  64 00'N 10 00'W  65 00'N 13 35'W  66 00'N 20 00'W	 
    #NORWEGIAN BASIN
    #  71 00'N 10 00'W  71 00'N 05 00'E  65 00'N 05 00'E  61 50'N 02 30'W  64 00'N 10 00'W
    #WEST NORTHERN SECTION
    #  65 00'N 27 30'W  65 00'N 40 00'W  55 00'N 40 00'W  55 00'N 27 30'W	
    #EAST NORTHERN SECTION
    #  65 00'N 27 30'W   55 00'N 27 30'W  55 00'N 15 00'W  62 30'N 15 00'W  63 40'N 18 00'W	 
    #WEST CENTRAL SECTION
    #  55 00'N 27 30'W  55 00'N 40 00'W  45 00'N 40 00'W  45 00'N 27 30'W	 
    #EAST CENTRAL SECTION
    #  55 00'N 15 00'W  55 00'N 27 30'W  45 00'N 27 30'W  45 00'N 15 00'W

    return shippingareas


##---
## calculation routines

def testShape(hs):
    """Tests for type of input array"""

    # expected inputs are (y,x), (t,y,x), (r,t,y,x)
    shapehs = len(np.shape(hs))
    if shapehs == 2:
        print('[INFO] Testing a 2D (x,y) array')
    elif shapehs == 3:
        print('[INFO] Testing a 3D (t,x,y) array')
    elif shapehs == 4:
        print('[INFO] Testing a 4D (realization,t,x,y) array')

    return shapehs


def seastatesP(hs, scale='douglas', ptype='occur'):
    """Generate seastate/wind force probability data from Hs/Wspd field
       Inputs:
        hs: input wave height or wind speed field
        scale: seastate/wind category scale (see set_seastates)
        ptype: probability type; occur (default) is probability of event in population,
                                 encounter is probability event is encountered in sample"""

    defseastates = set_seastates(scale=scale)

    # expected input shapes for data arrays are (y,x), (t,y,x), (r,t,y,x)
    shapehs = testShape(hs)

    if ptype.lower() == 'encounter':
        print('[INFO] Probability type is encounter')
    else:
        print('[INFO] Probability type is occur')

    pseastates = np.ma.zeros([len(defseastates),np.shape(hs)[-2],np.shape(hs)[-1]])

    for ind, seastate in enumerate(defseastates):
        print('[INFO] Testing for %s' %seastate)
        hsmin = defseastates[seastate][0]
        hsmax = defseastates[seastate][1]

        if shapehs == 2:
            # encounter and occurence probabilities will be the same
            pseastates[ind,:,:] = ((hs>=hsmin) & (hs<hsmax)).astype(np.int)
            pseastates[ind,:,:].mask = hs.mask
        if shapehs == 3:
            # encounter probabilities set to 1 if sea-state occurs along time axis
            if ptype.lower() == 'encounter':
                pseastates[ind,np.any((hs>=hsmin) & (hs<hsmax), axis=0)] = 1.0
            else:
                pseastates[ind,:,:] = np.ma.array(np.count_nonzero((hs>=hsmin) & (hs<hsmax), axis=0), dtype=float) / \
                                       np.float(np.shape(hs)[0])
            pseastates[ind,:,:].mask = hs[0,:,:].mask
        if shapehs == 4:
            # encounter probabilities defined by number of realizations containing event
            if ptype.lower() == 'encounter':
                tmp = np.ma.zeros([np.shape(hs)[0],np.shape(hs)[-2],np.shape(hs)[-1]])
                tmp[np.any((hs>=hsmin) & (hs<hsmax), axis=1)] = 1.0
                pseastates[ind,:,:] = np.ma.mean(tmp, axis=0)
            else:
                pseastates[ind,:,:] = np.ma.array(np.count_nonzero((hs>=hsmin) & (hs<hsmax), axis=tuple([0,1])), dtype=float) / \
                                       np.float(np.shape(hs)[0] * np.shape(hs)[1])
            pseastates[ind,:,:].mask = hs[0,0,:,:].mask

    return defseastates, pseastates


def seastate_from_pseastates(pseastates):
    """Pick the highest most likely seastate from probability information"""

    mlseastate = np.ma.ones([np.shape(pseastates)[1],np.shape(pseastates)[2]])
    cpseastate = np.ma.zeros([np.shape(pseastates)[1],np.shape(pseastates)[2]])
    for lp in range(np.shape(pseastates)[0]):
        mlseastate[pseastates[lp,:,:] >= cpseastate[:,:]] = lp
        cpseastate[pseastates[lp,:,:] >= cpseastate[:,:]] = pseastates[lp,pseastates[lp,:,:] >= cpseastate[:,:]]
    mlseastate.mask = pseastates[0,:,:].mask
    cpseastate.mask = pseastates[0,:,:].mask

    return mlseastate, cpseastate   


def numWaves(period, window=3600.0):
    """Return integer number of waves during window"""

    numwaves = np.ma.floor(window / period)
    numwaves[numwaves > window/3.0] = np.ceil(window/3.0)

    return numwaves


def ForristallH(hs, p, expected=False, rayleigh=False):
    """Return a wave height from the Forristall or Rayleigh distribution
       Based on:
         On the Statistical Distribution of Wave Heights in a Storm
         Forristall, G., JGR-Atmospheres, January 1978"""

    # Forristall constants
    alpha = 2.126
    beta = 8.42
    #Rayleigh distribution setting
    if rayleigh:
        alpha = 2.0
        beta  = 8.0

    if expected:
        # this is the expected value of all waves above prob threshold p
        theta = np.log(1.0 / (1.0-p))
        gamma = 0.5772
        h = (beta * theta)**(1.0 / alpha) * (1.0 +  gamma / (alpha * theta))
    else:
        h = ((np.log(1.0-p) * beta * -1.0) ** (1.0 / alpha) ) 

    # convert from Forristall's normalised value to real world
    h = h * hs / 4.0

    return h


def ForristallP(h, hs, rayleigh=False):
    """Return probability of wave height exceeding a given threshold
       based on background significant wave height and the Forristall
       or Rayleigh distribution
       Based on:
         On the Statistical Distribution of Wave Heights in a Storm
         Forristall, G., JGR-Atmospheres, January 1978"""

    # Forristall constants
    alpha = 2.126
    beta = 8.42
    #Rayleigh distribution setting
    if rayleigh:
        alpha = 2.0
        beta  = 8.0

    # Normalise wave heights
    h = h / (hs / 4.0)

    # Probability 
    p = np.exp(-1.0 * h**alpha / beta)

    return p


def calcHmax(hs, ptype='dist', p=0.99, tp=None, window=3600., expected=False, rayleigh=False):
    """Choose a type of Hmax calculation to perform:
       - direct probability from distribution (dist)
       - Hmax for number of waves in sample (sample)
       - avoid probability for number of waves in sample (avoid)

       Uses Forristall (default) or Rayleigh distribution"""

    if ptype.lower() == 'sample':
        numwaves = window / tp
        p = 1. - 1./numwaves        

    elif ptype.lower() == 'avoid':
        numwaves = window / tp
        p = p**(1./numwaves)

    hmax = ForristallH(hs, p, expected=expected, rayleigh=rayleigh)

    return p, hmax


def hmaxP(h, hs, numwaves=450, rayleigh=False, ptype='occur', tsteep=None, slim=0.11):
    """Calculate probability of encountering an extreme
       wave of height h within a sea state hs of n waves
       Inputs:
        h - threshold wave height for individual wave
        hs - background sea-state
        numwaves - number of waves encountered in sample period
        rayleigh - uses Rayleigh settings in Hmax calcs (default is Forristall)
        ptype: probability type; occur (default) is probability of event in population,
                                 encounter is probability event is encountered in sample
        tsteep: include period data in order to check whether waves are close to breaking
                seas limit"""

    # expected input shapes for data arrays are (y,x), (t,y,x), (r,t,y,x)
    shapehs = testShape(hs)

    if ptype.lower() == 'encounter':
        print('[INFO] Probability type is encounter')
    else:
        print('[INFO] Probability type is occur')

    phmaxtmp = ForristallP(h, hs, rayleigh=rayleigh)

    if tsteep is not None:
        print('[INFO] Including calculation of Hmax breaking wave probability')
        p, hmax   = calcHmax(hs, ptype='sample', tp=tsteep, window=3600., expected=False, rayleigh=rayleigh)
        psteep = sigSteepness(hmax, tsteep, depth=None)
        #print(np.max(hmax))
        #print(np.max(psteep))
        psteep[psteep >= slim] = 1.0
        psteep[psteep < slim]  = 0.0
        phmaxtmp = phmaxtmp * psteep

    if ptype.lower() == 'encounter':
        # probability of avoiding Hmax for multiple waves in sea-state
        pavoid = (1.0-phmaxtmp)**numwaves

        if shapehs == 3:
            # avoid across multiple time-steps (assumed independent)
            # integrate along time axis - assumed axis zero for deterministic
            pavoid = np.prod(pavoid, axis=0)
        elif shapehs == 4:
            # avoid across multiple time-steps (assumed independent)
            # integrate along time axis - assumed axis one for eps
            pavoid = np.prod(pavoid, axis=1)
            # avoid in all my eps scenarios
            # integrate along realization axis - assumed axis 0 for eps
            # also assume all scenarios equally likely - use mean
            pavoid = np.ma.mean(pavoid, axis=0)

        phmax = 1.0 - pavoid

    else:

        if shapehs == 3:
            # avoid across multiple time-steps (assumed independent)
            # integrate along time axis - assumed axis zero for deterministic
            phmax = np.ma.mean(phmaxtmp, axis=0)
        elif shapehs == 4:
            # integrate along time and realization axis - assumed axis 0 for eps
            # also assume all scenarios equally likely - use mean
            phmax = np.ma.mean(phmaxtmp, axis=(0,1))
    #print(np.max(phmax))

    return phmax


def sigSteepness(hs, tz, depth=None):
    """Calculate significant wave steepness"""

    # Deep water
    ssteep = 2.0 * np.pi * hs / (9.81 * tz**2.0)

    return ssteep


def steepP(hs, tz, depth=None, csteep=0.05, chs=None, ptype='occur'):
    """Calculate probabilities of high steepness wave conditions
       Additional use of chs allows a dangerous seas index similar
       to Savina and Lefevre, or IMA, to be applied
       Inputs:
        hs - significant wave height field
        tz - mean zero upcrossing period
        csteep - critical steepness threshold
        chs - critical significant wave height threshold
        ptype: probability type; occur (default) is probability of event in population,
                                 encounter is probability event is encountered in sample"""

    # expected input shapes for data arrays are (y,x), (t,y,x), (r,t,y,x)
    shapehs = testShape(hs)

    if ptype.lower() == 'encounter':
        print('[INFO] Probability type is encounter')
    else:
        print('[INFO] Probability type is occur')

    ssteep = sigSteepness(hs, tz, depth=depth)
    psteep = np.ma.zeros([np.shape(hs)[-2],np.shape(hs)[-1]])
    if shapehs == 2:
        # encounter and occurence probabilities will be the same
        if chs is not None:
            psteep[:,:] = ((ssteep > csteep) & (hs > chs)).astype(np.int)
        else:
            psteep[:,:] = (ssteep > csteep).astype(np.int)
        psteep[:,:].mask = hs.mask
    if shapehs == 3:
        if chs is not None:
            # encounter probabilities set to 1 if sea-state occurs along time axis
            if ptype.lower() == 'encounter':
                psteep[np.any((ssteep > csteep) & (hs > chs), axis=0)] = 1.0
            else:
                psteep[:,:] = np.ma.array(np.count_nonzero((ssteep > csteep) & (hs > chs), axis=0), dtype=float) / \
                                           np.float(np.shape(hs)[0])
        else:
            # encounter probabilities set to 1 if sea-state occurs along time axis
            if ptype.lower() == 'encounter':
                psteep[np.any(ssteep > csteep, axis=0)] = 1.0
            else:
                psteep[:,:] = np.ma.array(np.count_nonzero(ssteep > csteep, axis=0), dtype=float) / \
                                           np.float(np.shape(hs)[0])
        psteep = np.ma.masked_where(hs[0,:,:].mask, psteep)
    if shapehs == 4:
        if chs is not None:
            # encounter probabilities defined by number of realizations containing event
            if ptype.lower() == 'encounter':
                tmp = np.ma.zeros([np.shape(hs)[0],np.shape(hs)[-2],np.shape(hs)[-1]])
                tmp[np.any((ssteep > csteep) & (hs > chs), axis=1)] = 1.0
                psteep[:,:] = np.ma.mean(tmp, axis=0)
            else:
                psteep[:,:] = np.ma.array(np.count_nonzero((ssteep > csteep) & (hs > chs), axis=tuple([0,1])), dtype=float) / \
                                           np.float(np.shape(hs)[0] * np.shape(hs)[1])
        else:
            # encounter probabilities defined by number of realizations containing event
            if ptype.lower() == 'encounter':
                tmp = np.ma.zeros([np.shape(hs)[0],np.shape(hs)[-2],np.shape(hs)[-1]])
                tmp[np.any(ssteep > csteep, axis=1)] = 1.0
                psteep[:,:] = np.ma.mean(tmp, axis=0)
            else:
                psteep[:,:] = np.ma.array(np.count_nonzero(ssteep > csteep, axis=tuple([0,1])), dtype=float) / \
                                           np.float(np.shape(hs)[0] * np.shape(hs)[1])
        psteep = np.ma.masked_where(hs[0,0,:,:].mask, psteep)

    return psteep


def deltaZ(hs, thresh=0.2, hsthresh=2.5, tdelta=6.0, tstep=1.0, ptype='occur'):
    """Tests for increase of input parameter within time tdelta
       Used to test for rapid changes in sea-state following Toffoli et al., 2005
       Inputs:
         hs - wave parameter, e.g. significant wave height
         thresh - exceedence threshold for variable delta
         hsthresh - sets a threshold value for the variable maximum
         tdelta - time window over which to check for changes (hours)
         tstep - data time interval (hours)
         ptype: probability type; occur (default) is probability of event in population,
                                  encounter is probability event is encountered in sample"""

    # expected input shapes for data arrays are (y,x), (t,y,x), (r,t,y,x)
    shapehs = testShape(hs)
    if shapehs == 3:
        taxis = 0
    elif shapehs == 4:
        taxis = 1

    if ptype.lower() == 'encounter':
        print('[INFO] Probability type is encounter')
    else:
        print('[INFO] Probability type is occur')

    nvals = np.int(tdelta/tstep)

    pdeltaZ= np.ma.zeros([np.shape(hs)[-2],np.shape(hs)[-1]])
    if shapehs == 4:
        tmpdeltaZ= np.ma.zeros([np.shape(hs)[0],np.shape(hs)[-2],np.shape(hs)[-1]])
    # find events within windows of time-series
    if np.shape(hs)[taxis] >= nvals:
        print('[INFO] Testing %d windows of %d hours' %(np.shape(hs)[taxis]-nvals+1,tdelta))
        for iw in range(0,np.shape(hs)[taxis]-nvals+1):
            if shapehs == 3:
                print('window '+str(iw))
                tsmax = np.amax(hs[iw:iw+nvals,:,:], axis=taxis)
                idmax = np.argmax(hs[iw:iw+nvals,:,:], axis=taxis)
                tsmin = np.amin(hs[iw:iw+nvals,:,:], axis=taxis)
                idmin = np.argmin(hs[iw:iw+nvals,:,:], axis=taxis)
                print(tsmax)
            elif shapehs == 4:
                tsmax = np.amax(hs[:,iw:iw+nvals,:,:], axis=taxis)
                idmax = np.argmax(hs[:,iw:iw+nvals,:,:], axis=taxis)
                tsmin = np.amin(hs[:,iw:iw+nvals,:,:], axis=taxis)
                idmin = np.argmin(hs[:,iw:iw+nvals,:,:], axis=taxis)
            # check that window minimum occurs earlier than window maximum
            # there is a chance of missing some events with this method, but not if the window length is short
            tsmax[idmin >= idmax] = tsmin[idmin >= idmax]
            tsdelta = (tsmax - tsmin) / tsmin
            tsdelta[tsdelta < (1.0+thresh)] = 0.0
            tsdelta[tsdelta >= (1.0+thresh)] = 1.0
            tsmax[tsmax < hsthresh] = 0.0
            tsmax[tsmax >= hsthresh] = 1.0
            if shapehs == 3:
                pdeltaZ = pdeltaZ + tsmax * tsdelta
            elif shapehs == 4:
                tmpdeltaZ = tmpdeltaZ + tsmax * tsdelta
        if ptype.lower() == 'occur':
             # occurrence probabilities defined by number of windows containing event
            if shapehs == 3:
                pdeltaZ = pdeltaZ / np.float(np.shape(hs)[taxis]-nvals+1)
            elif shapehs == 4:
                tmpdeltaZ = tmpdeltaZ / np.float(np.shape(hs)[taxis]-nvals+1)
                pdeltaZ = np.sum(tmpdeltaZ,axis=0) / np.float(np.shape(hs)[0])
        elif ptype.lower() == 'encounter':
            if shapehs == 3:
                # encounter probabilities set to 1 if time-series contains event
                pdeltaZ[pdeltaZ > 0.0] = 1.0
            if shapehs == 4:
                # encounter probabilities defined by number of realizations containing event
                tmpdeltaZ[tmpdeltaZ > 0.0] = 1.0
                pdeltaZ = np.sum(tmpdeltaZ,axis=0) / np.float(np.shape(hs)[0])
        if shapehs == 3:
            pdeltaZ = np.ma.masked_where(hs[0,:,:].mask, pdeltaZ)
        elif shapehs == 4:
            pdeltaZ = np.ma.masked_where(hs[0,0,:,:].mask, pdeltaZ)
    else:
        print('[WARN] Input time-series too short for analysis with tdelta %d; returning zero probabilities' %np.int(tdelta))

    return pdeltaZ


def crossSea(hvals, dvals, hsthresh=2.5, dthresh=30.0, cratio=0.6, ptype='occur'):
    """Crossed-sea probability based on wave components
    follows Kohno, use direction separation of 30 and ratio of primary to secondary at 0.6"""

    # expected input shapes for data arrays are (y,x), (t,y,x), (r,t,y,x)
    shapehs = testShape(hvals[0])

    # populate summed, primary and secondary hs arrays
    hsp = np.zeros(np.shape(hvals[0]))
    dmp = np.zeros(np.shape(hvals[0]))
    hss = np.zeros(np.shape(hvals[0]))
    dms = np.zeros(np.shape(hvals[0]))
    for icmp in range(len(hvals)):
        dms[(hvals[icmp] > hsp) & (hsp > hss)] = dmp[(hvals[icmp] > hsp) & (hsp > hss)]
        hss[(hvals[icmp] > hsp) & (hsp > hss)] = hsp[(hvals[icmp] > hsp) & (hsp > hss)]
        dmp[hvals[icmp] > hsp]  = dvals[icmp][hvals[icmp] > hsp]
        hsp[hvals[icmp] > hsp]  = hvals[icmp][hvals[icmp] > hsp]
    sumhs = np.sqrt(hsp**2.0 + hss**2.0)

    # create 1/0 array for crossed sea, from component energy and direction difference
    pcmp = np.zeros(np.shape(hvals[0]))
    pcdm = np.zeros(np.shape(hvals[0]))
    pcmp[hsp*0.6 <= hss] = 1.0
    dirdiff = np.abs(dmp - dms)
    dirdiff[dirdiff > 180.0] = 360.0 - dirdiff[dirdiff > 180.0]
    pcdm[dirdiff > dthresh] = 1.0
    pxSea = pcmp * pcdm
    # only consider crossed sea that exceed the Hs threshold
    pxSea[sumhs < hsthresh] = 0.0

    if ptype.lower() == 'encounter':
        print('[INFO] Probability type is encounter')
        if shapehs == 3:
            pCrossSea = np.sum(pxSea, axis=0)
            pCrossSea[pCrossSea > 0.0] = 1.0
        elif shapehs == 4:
            pxSeaT = np.sum(pxSea, axis=1)
            pxSeaT[pxSeaT > 0.0] = 1.0
            pCrossSea = np.sum(pxSeaT, axis=0) / np.float(np.shape(hvals[0])[0])
    else:
        print('[INFO] Probability type is occur')
        if shapehs == 3:
            pCrossSea = np.sum(pxSea, axis=0) / np.float(np.shape(hvals[0])[0])
        elif shapehs == 4:
            pxSeaT = np.sum(pxSea, axis=1) / np.float(np.shape(hvals[0])[1])
            pCrossSea = np.sum(pxSeaT, axis=0) / np.float(np.shape(hvals[0])[0])
    if shapehs == 3:
        pCrossSea = np.ma.masked_where(hvals[0][0,:,:].mask, pCrossSea)
    elif shapehs == 4:
        pCrossSea = np.ma.masked_where(hvals[0][0,0,:,:].mask, pCrossSea)

    return pCrossSea
