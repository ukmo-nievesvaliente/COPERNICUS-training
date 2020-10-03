## module to define the irefrac refraction process

import numpy as np
import NearshoreAlgorithms as na

# estimate a wave height and saturated breaker depth for initial wave breaking
def CallGodaBreakHsInit( hsin, tpin, slope=0.01, tptotz=True ):

    # convert input tp to tz, tz seems to work better with Goda relationship
    if tptotz:
        tpin = tpin / 1.28  # based on Carter's JONSWAP calcs, a PM version would be ~1.4

    # apply the Goda breaker height estimate
    hsbkinit, dpsat = na.GodaBreakHsInit( hsin, tpin, slope=slope )

    return hsbkinit, dpsat


def RefracEqOff( hsin, drin, tpin, beachnorm, maxang, offdp, approslope, condepth=4.0, condepthbkest=True, verbose=False ):
    """Calculate an equivalent offshore wave height using direction filter, refraction factor and bottom friction estimate

       Expected inputs:
       hsin - deep water (model input) significant wave height; array
       drin - deep water (model input) wave direction (from); array
       tpin - deep water (model input) wave period (either tp or tz); array
       beachnorm - direction faced by beach normal; scalar or array with appropriate dimensions
       maxang - exposure half window angle for beach; scalar or array with appropriate dimensions
       offdp - depth of water offshore; scalar or array with appropriate dimensions
       approslope - estimated slope for bathymetry between offshore point and beach; scalar or array with appropriate dimensions

       Options:
       condepth - depth at beach boundary; scalar or array with appropriate dimensions
       condepthbktest - logical to request a second loop where condepth changes based on 1st guess breaker height

       Outputs:
       hsshwd - equivalent offshore wave height
       tpshwd - period for shoreward propagating waves
       newdir - revised offshore direction of approach relative to beach normal for shoreward propaating energy"""

    if verbose:
        print('Calculating equivalent offshore waves for ',hsin,tpin,drin,beachnorm,maxang)

    # calculate offshore wave direction relative to shore normal
    dirin  = na.BeachNormDirn(drin,beachnorm)

    # maximum angle adjustment based on ability of waves to refract - first guess using fixed inshore depth
    exang = na.RefracOff(tpin, maxang, condepth, offdepth=offdp)

    # calculate the shoreward propagating energy - first guess using fixed inshore depth
    hsshwd, newdir = na.SWardHs( hsin, dirin, maxang=exang )

    # set nearshore contour depth using first guess breaker height and repeat calculations for hsshwd
    if condepthbkest:
        condepth = np.maximum( 1.0, hsshwd*2.5 )
        if verbose:
            print('condepths revised based on breaker height estimate to ',condepth)
            print('re-calculating exang and hsshwd')

        # maximum angle adjustment based on ability of waves to refract
        exang = na.RefracOff(tpin, maxang, condepth, offdepth=offdp)

        # calculate the shoreward propagating energy
        hsshwd, newdir = na.SWardHs( hsin, dirin, maxang=exang )

    # adjust using refraction factor - stretching of wave rays offshore before turn
    hsshwd = hsshwd * na.RefracFactorSimple( np.minimum(dirin, newdir) )

    # adjust using bottom friction dissipation factor
    # this is not very sophisticated just now - lots of assumptions, but does include correction for angle of propagation
    hsshwd = hsshwd * na.JBFCheat( tpin, offdepth=offdp, indepth=condepth, slope=approslope, dirin=newdir, JGamma=0.038 )

    # assign values to tpshwd and newdir
    tpshwd = tpin
    if np.isscalar( hsshwd ):
        if hsshwd == 0.0:
            tpshwd = 0.0
            newdir = 0.0
    else:
        tpshwd[ np.where( hsshwd == 0.0 ) ] = 0.0
        newdir[ np.where( hsshwd == 0.0 ) ] = 0.0

    return hsshwd, tpshwd, newdir


# call the Refrac routine
def CallRefracEqOffComponents( hsin, drin, tpin, beachnorm, maxang, offdp=20.0, approslope=0.005, condepth=4.0, condepthbkest=True, linsh=False ):
    """Call the IRefrac routine to calculate equivalent offshore wave height data if given a multi-component array.

       Expected inputs:
       hsin - deep water (model input) significant wave height; array, last dimension is number of components
       drin - deep water (model input) wave direction (from); array, last dimension is number of components
       tpin - deep water (model input) wave period (either tp or tz); array, last dimension is number of components
       beachnorm - direction faced by beach normal; array or scalar
       maxang - exposure half window angle for beach; array or scalar
       offdp - depth of water offshore; array or scalar
       approslope - estimated slope for bathymetry between offshore point and beach; array or scalar

       Options:
       condepth - depth at beach boundary; scalar for first guess
       condepthbktest - logical to request a second loop where condepth changes based on 1st guess breaker height

       Outputs:
       hsshwd - equivalent offshore wave height
       tpshwd - period for shoreward propagating waves
       newdir - revised offshore direction of approach relative to beach normal for shoreward propaating energy"""

    # run the routine to calculate equivalent offshore wave height for all times and components
    hsshwd_cmp, tpshwd_cmp, drshwd_cmp = RefracEqOff( hsin, drin, tpin, beachnorm, maxang, offdp, approslope, condepth, condepthbkest=condepthbkest, verbose=False )

    # calculate total wave field from multiple components array

    # set up arrays for the total wave field values
    myshape = np.shape( hsshwd_cmp )
    hsshwd = np.zeros( myshape[0:-1] )
    tpshwd = np.zeros( myshape[0:-1] )
    drshwd = np.zeros( myshape[0:-1] )
    # create an indexing list for the time/site parts of the array
    indices1 = []
    for x in myshape[:-1]: indices1.append( slice(0,x) )

    # loop over components
    for lp in range( myshape[-1] ):
        # create an indexing tuple including the component index
        indices = indices1[:]
        indices.append(lp)
        indices = tuple(indices)
        # sum up the moments data
        hsshwd = hsshwd + hsshwd_cmp[indices]**2.0
        tpshwd = tpshwd + hsshwd_cmp[indices]**2.0 * tpshwd_cmp[indices]
        drshwd = drshwd + hsshwd_cmp[indices]**2.0 * drshwd_cmp[indices]
        
    # finalise arrays
    tpshwd[ np.where(hsshwd>0.0) ] = tpshwd[ np.where(hsshwd>0.0) ] / hsshwd[ np.where(hsshwd>0.0) ]
    drshwd[ np.where(hsshwd>0.0) ] = drshwd[ np.where(hsshwd>0.0) ] / hsshwd[ np.where(hsshwd>0.0) ]
    hsshwd = np.sqrt(hsshwd)

    return hsshwd, tpshwd, drshwd


def CallRefracSurf_RNLI( hsoff, diroff, tpoff, beach_normal, beach_half_window, beach_slope, approach_slope, depthoff=20.0, tptotz=True ):
    """Call the irefrac and isurf algorithms needed for RNLI data feeds.

       Expected inputs:
       hsoff[times,sites,comps]             - deep water (model input) significant wave height
       diroff[times,sites,comps]            - deep water (model input) wave direction (from)
       tpoff[times,sites,comps]             - deep water (model input) wave period (either tp or tz)
       beach_normal[sites] - direction faced by beach normal (e.g. west facing beach uses 270.0)
       beach_half_window[sites] - exposure angle for beach (valid range 5.0 to 85.0 degrees)
       beach_slope[sites] - local beach slope (will use tidal variation in later versions, valid range 0.001 to 0.1)
       approach_slope[sites] - estimated slope for bathymetry between offshore point and beach (valid range 0.001 to 0.1)
       depthoff[sites] - depth of water for model input point (defaults to deep water, valid minimum 10.0)
       tptotz [sites] - tp to tz conversion used in surf call as it seems to work better than using tp with Goda's algorithm

       Outputs:
       hlo    - low range breaker height
       hhi    - high range breaker height
       tpshwd - period used in breaking wave calcs
       ibn    - Irribarren number
       bktype - breaker type"""

    # check that window and slope data are in valid ranges as specified above
    beach_half_window = np.maximum( 5.0, beach_half_window )
    beach_half_window = np.minimum( 85.0, beach_half_window )
    beach_slope =  np.maximum( 0.001, beach_slope )   
    beach_slope =  np.minimum( 0.1, beach_slope )   
    approach_slope =  np.maximum( 0.001, approach_slope )   
    approach_slope =  np.minimum( 0.1, approach_slope )   

    # 1. calculate equivalent offshore hs for breaker calculations

    # a. ensure the beach information arrays are the correct shape - simplest method is to inflate arrays to same shape as wave inputs
    # exception when inputs are scalar - presently using beachnorm as test for this
    if not np.isscalar( beach_normal ):
        myshape = np.shape( hsoff )
        beach_normal_tmp = np.zeros( myshape )
        beach_half_window_tmp = np.zeros( myshape )
        depthoff_tmp = np.zeros( myshape )
        approach_slope_tmp = np.zeros( myshape )
        for lp in range( myshape[1] ):
            beach_normal_tmp[:,lp,:] = beach_normal[lp]
            beach_half_window_tmp[:,lp,:] = beach_half_window[lp]
            depthoff_tmp[:,lp,:] = depthoff[lp]
            approach_slope_tmp[:,lp,:] = approach_slope[lp]
    else:
        beach_normal_tmp = beach_normal
        beach_half_window_tmp = beach_half_window
        depthoff_tmp = depthoff
        approach_slope_tmp = approach_slope

    # b. call the refrac routine
    hsshwd, tpshwd, reldir = CallRefracEqOffComponents( hsoff, diroff, tpoff, beach_normal_tmp, beach_half_window_tmp, offdp=depthoff_tmp, approslope=approach_slope_tmp, condepth=4.0, condepthbkest=True, linsh=False )

    # 2. calculate the wave height at breaker intiation using Goda's algorithm

    # a. ensure the beach information arrays are the correct shape - simplest method is to inflate arrays to same shape as wave inputs
    # exception when inputs are scalar
    if not np.isscalar( beach_slope ):
        myshape = np.shape( hsshwd )
        beach_slope_tmp = np.zeros( myshape )
        for lp in range( myshape[-1] ):
            beach_slope_tmp[:,lp] = beach_slope[lp]
    else:
        beach_slope_tmp = beach_slope

    # b. call the Goda routine
    hsbkinit, dpsat = CallGodaBreakHsInit( hsshwd, tpshwd, slope=beach_slope_tmp, tptotz=tptotz )

    # c. estimate the hlo and hhi values based on Rayleigh distribution
    hlo = na.ForristallHs(hsbkinit, 0.67, rayleigh=True)
    hhi = na.ForristallHs(hsbkinit, 0.9, rayleigh=True)

    # d. calculate the Irribarren number and breaker type
    ibn, bktype = na.Iribarren(hsshwd, tpshwd, beach_slope_tmp, tptotz=tptotz, iboff=True)

    # add the output arrays to dictionary
    surfdata = {'shorewardHs':hsshwd, 'shorewardT':tpshwd, 'relativeDirn':reldir,
                'breakerHs':hsbkinit, 'saturatedDepth':dpsat, 'Hb1in3':hlo, 'Hb1in10':hhi,
                'iribarrenNumber':ibn, 'breakerType':bktype}

    # return the output arrays
    return surfdata


def CallRip_RNLI_notide( hsshwd, tpshwd, fwfac=10.0, hsbkinit=None, return_all=False ):
    """Call the rip algorithms needed for RNLI data feeds.
    """

    # calculate wave factor
    fwave = na.CalcFwave( hsshwd, tpshwd, fwfac=10.0, hsbkinit=hsbkinit)

    # calculate rip overall
    ripind = na.CalcRip( fwave, ftide=None, wspd=None, beachtype=None )
        
    if return_all:
        return fwave, ripind    
    else:
        return ripind


def CallRip_RNLI_tidal( beachtype, fctide, tideseries, hsshwd, tpshwd, fwfac=10.0, hsbkinit=None, dpsat=None, wspd=None, return_all=False ):
    """Call the rip algorithms needed for RNLI data feeds.
    """

    # a. ensure the beach information arrays are the correct shape - simplest method is to inflate arrays to same shape as wave inputs
    # exception when inputs are scalar
    if not np.isscalar( beachtype ):
        myshape = np.shape( hsshwd )
        beach_type_tmp = np.zeros( myshape )
        for lp in range( myshape[-1] ):
            beach_type_tmp[:,lp] = beachtype[lp]
    else:
        beach_type_tmp = beachtype

    # calculate wave factor
    fwave = na.CalcFwave( hsshwd, tpshwd, fwfac=10.0, hsbkinit=hsbkinit)

    # calculate tide factor
    ftide = na.CalcFtide( fctide, tideseries, dpsat=dpsat )
    #ftide = np.array( hsshwd>-1.0, dtype=bool )

    # calculate rip overall
    ripind = na.CalcRip( fwave, beachtype=beach_type_tmp, ftide=ftide, wspd=wspd )
        
    if return_all:
        return fwave, ftide, ripind    
    else:
        return ripind
