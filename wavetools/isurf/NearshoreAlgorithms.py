import numpy as np
from scipy.signal import argrelextrema

def CheckScalar(var):
    """Check for scalar vs array input"""

    var = np.asarray(var)  # elevates list, tuple, etc to a numpy array.
    return_scalar = False
    if np.ndim(var) == 0:   # safer than np.isscalar
        return_scalar = True
        var = var[np.newaxis] # make one-d

    return var, return_scalar


def BeachNormDirn(dirin, beachnorm):
    """Compute an angle of approach relative to beach normal.

       Expected inputs:
       dirin - direction (from) of waves; array
       beachnorm - direction toward which beach normal faces; scalar or array with appropriate dimensions

       Outputs:
       angle - absolute angle of wave approach relative to shore normal"""

    # check for scalar vs array input
    dirin, return_scalar = CheckScalar(dirin) 

    angle = np.abs(dirin - beachnorm)
    angle[np.where(angle > 180.0)] = 360.0 - angle[np.where(angle > 180.0)]

    if return_scalar:
        return angle.item()
    return angle


def RefracOff(t, inang, indepth, offdepth=None):
    """Calculate offshore angle of propagation based on nearshore angle of approach.

       Expected inputs:
       t - period; array
       inang - nearshore angle of approach relative to shore normal; scalar or array with appropriate dimensions
       indepth - nearshore depth; scalar or array with appropriate dimensions 
       offdepth - offshore depth; scalar or array with appropriate dimensions

       Outputs:
       offang - offshore angle of approach relative to shore normal"""

    # check for scalar vs array input
    t, return_scalar = CheckScalar(t) 

    # calculate wavelengths using fenton
    wvlin = WVLFenton(t, indepth)
    if offdepth is not None:
        wvloff = WVLFenton(t, offdepth)
    else:
        wvloff = 9.81 * t**2.0 / (2.0*np.pi)

    # offshore to inshore wavelength ratio
    wvlrat = np.ones(np.shape(t))
    wvlrat[np.where(t>0.5)] = wvloff[np.where(t>0.5)] / wvlin[np.where(t>0.5)]

    # refraction angles
    offang = wvlrat * np.sin(np.pi*inang/180.0)
    offang[np.where(offang>1.0)] = 1.0
    offang = np.arcsin(offang)
    offang = offang * 180.0 / np.pi

    offang = inang + 0.3 * (offang - inang) # reduced limiter vs qrefrac as integrating over frequencies

    if return_scalar:
        return offang.item()
    else:
        return offang

    
def RefracFactorSimple(offang, s0=1.0):
    """Estimate a refraction factor using ray spreading at point where waves start to turn.
       See wave ray diagram in tucker and pitt p297.

       Expected inputs:
       offang - angle of approach to shore normal

       Outputs:
       krefrac - refraction factor"""

    salong = 1.0 / np.cos( offang * np.pi / 180.0)
    krefrac = np.sqrt( s0 / salong )

    return krefrac


def SWardHs(hs, inang, maxang=None):
    """Algorithm to estimate wave energy likely to be transmitted shoreward from the offshore zone.
       Calculations in this algorithm assume that directional spread is represented by a cosine squared function.

       Expected inputs (numpy arrays):
       hs - significant wave height offshore, array
       inang - angle of the offshore waves relative to the beach shore normal; scalar or array with appropriate dimensions 
       maxang - half-window exposure angle for wave energy relative to shore normal; scalar or array with appropriate dimensions

       Outputs:
       hsdir  - wave height for energy transmitted shoreward
       newdir - direction of energy transmitted shoreward, relative to shore normal"""

    # check for scalar vs array input
    hs, return_scalar = CheckScalar(hs) 

    # convert input angle degrees to radians
    inangrad = inang * np.pi / 180.0

    # use the half-window exposure angle to set a max angle for wave energy
    # reaching an emabayed beach
    if maxang is None:
        angextent = 85.0 * np.pi / 180.0
    else:
        angextent = maxang * np.pi / 180.0

    # set negative and positive extent angles
    negextent = np.maximum(-1.0*angextent, -1.0*angextent + inangrad)
    posextent = np.minimum(angextent, angextent + inangrad)

    # integral at positve extent for energy calculation
    m0pos = 0.5 * posextent + 0.25 * np.sin(posextent * 2.0) # zeroth moment - energy in spectrum
    m1pos  = 0.25 * posextent**2.0 + 0.25 * posextent * np.sin(2.0 * posextent) + 0.125 * np.cos(2.0 * posextent) # 1st moment for direction calc

    # integral at negative extent for energy calculation
    m0neg = 0.5 * negextent + 0.25 * np.sin(negextent * 2.0) # zeroth moment - energy in spectrum
    m1neg  = 0.25 * negextent**2.0 + 0.25 * negextent * np.sin(2.0 * negextent) + 0.125 * np.cos(2.0 * negextent) # 1st moment for direction calc

    # calculate the integral
    cs2ang   = m0pos - m0neg # zeroth moment - energy in spectrum
    m1ang    = m1pos - m1neg # 1st moment for direction calc

    # overall energy calculation for a full cosine^2 spread spectrum
    cs2tot = np.pi / 2.0

    # calculate energy factor and hs from energy in the zeroth moment integrals
    hsdir = np.zeros(np.shape(hs))
    hsdir[np.where(cs2ang>0.0)] = hs[np.where(cs2ang>0.0)] * np.sqrt(cs2ang[ np.where(cs2ang>0.0)] / cs2tot)

    # revise direction based on proportion of energy left
    newdir = np.zeros(np.shape(hs))
    newdir[np.where(cs2ang>0.0)] = m1ang[np.where(cs2ang>0.0)] / cs2ang[np.where(cs2ang>0.0)]
    newdir[np.where(cs2ang>0.0)] = inang[np.where(cs2ang>0.0)] - newdir[np.where(cs2ang>0.0)] * 180.0 / np.pi

    if return_scalar:
        return hsdir.item(), newdir.item()
    else:
        return hsdir, newdir


def GodaBreakHsInit(hs, t, slope=0.01):
    """Algorithm to apply Goda formulation for prediction of significant wave height at onset of breaking.

       Expected inputs (numpy arrays):
       hs - offshore wave height (assumed deep water); array
       t  - offshore wave period (assumed deep water, the algorithm seems to work better with tz than tp); array
       slope - beach slope; scalar or array with appropriate dimensions 

       Outputs:
       hsd   - wave height at breakpoint
       dpsat - estimated depth of water in which wave breaking is saturated"""

    # check for scalar vs array input, both hs and t should be same type
    hs, return_scalar = CheckScalar(hs) 
    t,  return_scalar = CheckScalar(t) 

    # set up output arrays
    hsd   = np.zeros(np.shape(hs))
    dpsat = np.zeros(np.shape(hs))

    # create missing data arrays for next set of calcs
    # assume small waves breaking at shoreline if hs or t less than lower limits
    hsma = np.ma.masked_less(hs, 0.05)
    tma  = np.ma.masked_less(t, 1.0)

    # calculate offshore wavelength
    wvl = 9.81 * tma**2.0 / (2.0 * np.pi)

    # breaking is defined by goda coefficient b for Hs
    beta0   = 0.028 * (hsma/wvl)**(-0.38) * np.exp(20.0 * slope**1.5)
    beta1   = 0.52 * np.exp(4.2*slope)
    betamax = 0.32*(hsma/wvl)**(-0.29)*np.exp(2.4*slope)
    betamax = np.maximum(0.92, betamax)

    # calculate Hs according to Goda's equation 3.25
    hsdma = betamax * hsma

    # calculate depth at which this value intersects with saturated break point
    dpsatma = (hsdma - beta0 * hsma) / beta1

    hsd[np.ma.MaskedArray.nonzero(hsdma)] = hsdma[np.ma.MaskedArray.nonzero(hsdma)]
    dpsat[np.ma.MaskedArray.nonzero(hsdma)] = dpsatma[np.ma.MaskedArray.nonzero(hsdma)] 

    if return_scalar:
        return hsd.item(), dpsat.item()
    else:
        return hsd, dpsat


def WVLFenton(t, depth, return_deptype=False):
    """Calculate Fentons wavelength for intermediate wave depths.

       Expected inputs:
       t     - period; array
       depth - depth; scalar or array with appropriate dimensions

       Outputs:
       wvint - wavelength value
       deptype - water depth type, 0:shallow, 1:intermediate, 2:deep"""

    # check for scalar vs array input
    t, return_scalar = CheckScalar(t) 

    # set up the output arrays
    wvint   = np.zeros(np.shape(t))
    deptype = np.ones(np.shape(t), dtype=int)

    # use missing data array for calcs to ensure the data is masked when dividing by small numbers
    tma = np.ma.masked_less(t, 0.5)

    # calculate the shallow, deep and intermediate wavelength options
    wlshallow = np.sqrt(9.81 * depth) * tma
    wrshallow = depth / wlshallow
    wldeep    = 9.81 * tma**2.0 / (2.0 * np.pi)
    wrdeep    = depth / wldeep
    wlint     = wldeep * np.tanh((2.0*np.pi*depth/wldeep)**0.75)**(2.0/3.0)

    # put the correct option into the wvint and deptype arrays
    wvint[np.ma.MaskedArray.nonzero(wlint)] = wlint[np.ma.MaskedArray.nonzero(wlint)]
    wvint[np.where(wrshallow < 0.05)] = wlshallow[np.where(wrshallow < 0.05)]
    deptype[np.where(wrshallow < 0.05)] = 0
    wvint[np.where(wrdeep > 0.5)] = wldeep[np.where(wrdeep > 0.5)]
    deptype[np.where(wrdeep > 0.5)] = 2

    if return_deptype:
        if return_scalar:
            return wvint.item(), deptype.item()
        else:
            return wvint, deptype
    else:
        if return_scalar:
            return wvint.item()
        else:
            return wvint


def CgEstimate(t, depth):
    """Calculate group speed using Fentons wavelength for intermediate wave depths.
       Expected inputs:
       t     - period; array
       depth - depth; scalar or array with appropriate dimensions

       Outputs:
       cg - wave group speed value
       cr - group/phase speed ratio"""

    # check for scalar vs array input
    t, return_scalar = CheckScalar(t) 

    # set output arrays
    cg = np.zeros(np.shape(t))
    cr = np.ones(np.shape(t))

    wvint, deptype = WVLFenton(t, depth, return_deptype=True)

    # assign speeds to arrays based on deptype

    # shallow water
    cg[np.where(deptype==0)] = wvint[np.where(deptype==0) ] / t[ np.where(deptype==0)]
    cr[np.where(deptype==0)] = 1.0

    # test deep water
    cg[np.where(deptype==2)] = wvint[np.where(deptype==2) ] / t[ np.where(deptype==2)] * 0.5
    cr[np.where(deptype==2)] = 0.5

    # intermediate water
    c = wvint / t
    cg_tmp = (c/2.0) * (1 + (4.0*np.pi*depth/wvint) / (np.sinh( 4.0*np.pi*depth/wvint )))
    cr_tmp = cg_tmp / c
    cg[np.where(deptype==1)] = cg_tmp[np.where(deptype==1)]
    cr[np.where(deptype==1)] = cr_tmp[np.where(deptype==1)]

    if return_scalar:
        return cg.item(), cr.item()
    else:
        return cg, cr


def JBFCheat(t, offdepth=20.0, indepth=4.0, slope=0.005, dirin=0.0, JGamma = 0.038, maxdist=10000.0, iterval=20.0, verbose=False):
    """Algorithm to apply a correction factor based on bottom friction dissipation.
       This is a cheat that assumes we are in transition zone from deep to shallow waters and uses Hasselmann's 
       JONSWAP BF coefficient defaults to 0.038 for swell - could be 0.067 for wind-sea.

       Expected inputs:
       t - wave period; array
       offdepth - offshore depth value; scalar or array with appropriate dimensions
       indepth - inshore depth value; scalar or array with appropriate dimensions
       slope - estimated slope on approach to beach; scalar or array with appropriate dimensions
       dirin - angle of waves approach to beach; scalar or array with appropriate dimensions

       Tuning parameters:
       JGamma - JONSWAP bottom friction coefficient; scalar
       maxdist - maximum permissible distance from beach to offshore location; scalar
       iterval - number of iteractions used to derive bottom friction factor (simulates cell stepping in SWAN); scalar

       Outputs:
       bffac - bottom friction factor to apply to Hs"""

    # check for scalar vs array input
    t, return_scalar = CheckScalar(t) 

    # some checks to make sure we don't use stupid numbers
    slope = np.maximum(0.001, slope) # min slope in in 1000
    t = np.maximum(2.5, t) # min period 2.5 seconds
    indepth  = np.maximum(1.0, indepth) # min inshore depth 1m
    offdepth = np.minimum(10.0, offdepth) # min offshore depth 10m (standard model minimum)
    depthdiff_direct = offdepth - indepth # array for depth checks
    depthdiff_slopemax = slope * maxdist # second array for depth checks, this uses slope times a maximum dissipation distance
    depthdiff = np.minimum(depthdiff_direct, depthdiff_slopemax) # where slopemax value is less than the direct calc use slopemax
    depthdiff = np.maximum(0.5, depthdiff) # remove any really small/negative values
    offdepth  = indepth + depthdiff # replace offdepth with the checking array - if inputs are ok there should be no difference!

    # travel time for wave approach to the beach is determined using mean offshore-shallow water cg and the offshore slope
    cgoff, croff = CgEstimate(t, offdepth)
    cgshl, crshl = CgEstimate(t, indepth)
    cgint, crint = CgEstimate(t, (offdepth+indepth)/2.0) #introduce an intermediate depth for where cg very variable

    cgmean = (cgshl + cgoff + cgint) / 3.0
    crmean = (crshl + croff + crint) / 3.0
    dpmean = (offdepth * (cgshl+cgint/2.0) + indepth * (cgoff+cgint/2.0)) / (cgshl + cgoff + cgint) # weight mean depth using wave speed
    bffac = 2.0 * JGamma * np.maximum(0.0,(crmean - 0.5))  * ((offdepth - indepth) / (cgmean * slope )) / (9.81 * dpmean)

    # apply directional correction
    # phase speed ratio helps correct for curvature for longer period waves turning in more quickly
    aninfac = 1.0 / np.cos(np.minimum((cgshl*croff/(cgoff*crshl)), 1.0) * dirin * np.pi / 180.0)  
    bffac   = bffac * np.minimum(5.0, aninfac)

    # finalise bffac, use iterval to simulate grid cell by grid cell energy reduction
    # rather than integrating by time over full profile
    bffac   = (1.0 - bffac/iterval)**iterval
    bffac = np.sqrt(np.maximum(0.0, bffac)) # conversion as we multiply hs by bffac, not energy

    if return_scalar:
        return bffac.item()
    else:
        return bffac


# return a wave height from the Forristall distribution
def ForristallHs(hs, p, rayleigh=False):
    """Return a wave height from the Forristall distribution.

       Expected inputs:
       hs - significant wave height; scalar or array
       p (scalar) - percentile of distribution (e.g. 0.99 has 1% chance of being exceeded)
       rayleigh - use the Rayleigh distribution instead of Forristall

       Outputs:
       h - wave height"""

    # Forristall constants
    alpha = 2.126
    beta = 8.42

    #Rayleigh distribution setting
    if rayleigh:
        alpha = 2.0
        beta  = 8.0

    h = ((np.log(1.0-p) * beta * -1.0) ** (1.0 / alpha) ) * hs / 4.0 

    return h


# calculate the Iribarren number
def Iribarren(hs, t, slope, tptotz=False, iboff=True):
    """Calculate the Irribarren number and breaker type based on either deep water or breaking waves.

       Expected inputs (numpy arrays):
       hs - wave height (deep water or breaking, dependant on iboff, True if deep water); array
       t  - wave period (tptotz will convert and input peak period to zero-upcrossing); array
       slope - beach slope; scalar or array with appropriate dimensions 

       Outputs (numpy arrays):
       ibnum[time,site]  - Iribarren number
       bktype[time,site] - breaker type
         Returned breaker type numbers are as follows:
         0 - spilling
         1 - plunging
         2 - surging or collapsing"""

    # check for scalar vs array input, both hs and t should be same type
    hs, return_scalar = CheckScalar(hs) 
    t,  return_scalar = CheckScalar(t) 

    # convert and input tp to a tz value
    if tptotz:
        t = t / 1.28  # based on Carter's JONSWAP calcs, a PM version would be ~1.4

    # use missing data array for calcs to ensure the data is masked when dividing by small numbers
    hsma = np.ma.masked_less(hs, 0.01)
    tma  = np.ma.masked_less(t, 0.5)

    # iribarren relationship
    ibnum = np.zeros(np.shape(hs))
    wvloff = 9.81 * tma**2.0 / (2.0 * np.pi)
    ibnum_ma = slope / np.sqrt(hsma / wvloff)
    ibnum[np.ma.MaskedArray.nonzero(ibnum_ma)] = ibnum_ma[np.ma.MaskedArray.nonzero(ibnum_ma)]

    # select breaker type
    ibtypes = [0.5, 3.3] # ibnum in deep water
    if iboff == False:
        ibtypes = [0.4, 2.0] # ibnum at break-point
    bktype = np.zeros(np.shape(ibnum), dtype=int)
    bktype[np.where(ibnum > ibtypes[0])] = 1    
    bktype[np.where(ibnum > ibtypes[1])] = 2    

    if return_scalar:
        return ibnum.item(), bktype.item()
    else:
        return ibnum, bktype


def CalcFwave(hs, t, fwfac=10.0, hsbkinit=None):
    """Calculates Fwave for rip indicator.

       Expected inputs:
       hs - significant wave height; array
       t - wave period; array
       
       Optional inputs
       fwfac - scaling value for fwave; scalar or array
       hsbkinit - breaking wave height used as an override for low  wave fwave; array

       Outputs:
       fwave - wave factor for rip indicator calculations"""

    # check hs and t have array form for calcs
    hs, return_scalar = CheckScalar(hs)
    t, return_scalar = CheckScalar(t)

    # calculate fwave
    fwave = hs * t / fwfac

    # apply reduction if hsbkinit provided to routine
    if hsbkinit is not None:
        fwave[hsbkinit < 0.25] = 0.0

    if return_scalar:
        return fwave.item()
    else:
        return fwave


def CalcFtide(fctide, tideseries, dpsat=0.5, dpsatscl=0.5):
    """Generates boolean value to determine state of tide wrt position of bar-rip   
       morphology. 

    Expected inputs:
    fctimes - tide values at times for prediction; array [time,sites]
    tideseries - full tidal time-series, used to calculate mean low water; array [time,sites]

    Optional inputs:
    dpsat - value to correct for wave height at breakpoint; scalar or array with appropriate dimensions [time,sites]

    Outputs:
    ftide - boolean, true when tide elevation on low water morphology; array [time,sites]"""

    # find minima and calculate mean low water value for each site
    lwmean = np.zeros(np.shape(tideseries)[1])
    for lp in range(np.shape(tideseries)[1]):
        mins = argrelextrema(tideseries[:,lp], np.less, axis=0)
        lwtest = tideseries[mins,lp]
        lwmean[lp] = np.mean(lwtest[ lwtest<-0.2 ]) # used in order to ignore double high waters?
    
    # calculate the difference from mean low water in forecast series
    tidediff = fctide - lwmean

    # add the wave effect
    # dpsat is reduced by a scale factor to represent depth where majority of waves are breaking
    tidediff = tidediff - dpsat * dpsatscl

    # ftide is true where tidediff is negative, i.e. wave breaking occurs below mean lw
    ftide = np.array(tidediff<0.0, dtype=bool)

    return ftide


def CalcRip(fwave, beachtype=None, ftide=None, wspd=None):
    """Calculates the rip indicator value (1-5) based on fwave and ftide.

       Expected inputs:
       fwave - wave factor ( height, period combination ); array

       Optional inputs:
       beachtype - beach type (numeric); scalar or array with same dimensions as fwave
       ftide - tide factor (boolean, must be provided with beachtype); array
       wspd - wind speed in metres per second; array

       Outputs:
       ripind - rip risk indicator (1 to 5)
       1 - negligible risk; 2 - moderate wave risk; 3 - moderate waves on bar-rip profile;
       4 - moderate waves and wind on bar-rip profile; 5 high wave risk

       Note on beach types:
       These values follow the beach classification in Scott et al (2014) Mar. Geol.
       """

    # category values for beaches with low tide bar-rip morphology
    rip_classes = np.array([4,5,6,7,8])

    # set hardwired thresholds for fwave and wind speed
    fwavelo = 0.5
    fwavehi = 1.5
    wshi    = 7.0

    # check that expected inputs are arrays 
    fwave, return_scalar = CheckScalar(fwave)
    wspd, return_scalar = CheckScalar(wspd)

    # default rip indicator is 2
    ripind = np.ones(np.shape(fwave), dtype=int) * 2

    # if the beachtype is in the low-tide bar-rip category test for categories 3 and 4
    if beachtype is not None:
        # set up a boolean array with beach type data
        if np.isscalar(beachtype):
            rip_beach = np.zeros(np.shape(fwave), dtype=bool)
            if beachtype in rip_classes:
                rip_beach[:] = True
        else:
           # assume input beachtype array is same shape as fwave 
            beachtype1d = np.ravel(beachtype)
            rip_beach1d = np.in1d(beachtype1d, rip_classes)
            rip_beach = np.reshape(rip_beach1d, np.shape(fwave))

        # assign values to temporary low tide rip indicator array
        ripind_ltr = np.ones(np.shape(fwave), dtype=int) * 2
        ripind_ltr[np.where(ftide)] = 3
        # update the array with type 4 where needed
        if wspd is not None:
            ripind_ltr[np.where(np.logical_and(ftide,wspd>wshi))] = 4

        # put the low tide rip data into the main indicator array
        ripind[np.where(rip_beach)] = ripind_ltr[np.where(rip_beach)]

    # low wave case - beach type and state of tide is irrelevant
    ripind[np.where(fwave<fwavelo)] = 1

    # high wave case - beach type and state of tide is irrelevant
    ripind[np.where(fwave>fwavehi)] = 5
    
    if return_scalar:
        return ripind.item()
    else:
        return ripind
