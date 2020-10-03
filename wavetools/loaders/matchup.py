import numpy as np
from scipy.spatial import cKDTree

# functions for matchup, using varGrid object

def sea_points_only_grid(varGrid):
    """Create 1D seapoints only lat, lon, plus index arrays
       from 2D input lat, lon and field arrays.
       The returned indices array has dimensions (seapoints,2)"""

    print("[INFO] Reducing 2D lat-lon grid to seapoints only")

    # reshape x-y parts of input variable array to 1D
    hs = varGrid.data
    nlons = len(varGrid.glons)
    nlats = len(varGrid.glats)
    if varGrid.fctype == 'deterministic':
        hstmp = np.reshape(hs[0,:,:], [nlons*nlats])
    else:
        hstmp = np.reshape(hs[0,0,:,:], [nlons*nlats])

    latstmp = np.empty(nlons*nlats)
    lonstmp = np.empty(nlons*nlats)
    xtmp    = np.empty(nlons*nlats, dtype=int)
    ytmp    = np.empty(nlons*nlats, dtype=int)
    for iy in range(nlats):
        ix = iy * nlons
        latstmp[ix:ix+nlons] = varGrid.glats[iy]
        lonstmp[ix:ix+nlons] = varGrid.glons
        xtmp[ix:ix+nlons]    = np.arange(nlons,dtype=int)
        ytmp[ix:ix+nlons]    = iy

    # determine seapoints using mask from variable data
    latssp = latstmp[np.where(hstmp.mask == False)]
    lonssp = lonstmp[np.where(hstmp.mask == False)]
    xssp   = xtmp[np.where(hstmp.mask == False)]
    yssp   = ytmp[np.where(hstmp.mask == False)]
    indices = np.transpose(np.array([xssp, yssp]))    

    return latssp, lonssp, indices


def nearest_indices(mlon, mlat, mindices, lon, lat, rngtol=10000., return_compressed=True):
    """ Function to calculate wave model grid indices
        for the point locations specified in lats/lons.
        Max dist is given in km and converted to degrees in script.
        Two return options are available: compressed lists of requested and found indices,
        or a list of found indices where missing sites are set to -1.
    """

    # max distance for grid cell searching
    max_dist = (rngtol / 1000.0) / (1.853 * 60.)
    print("[INFO] Max search radius is %.3f degrees" % max_dist)

    mlon[mlon < -180] += 360.
    mlon[mlon > 180] -= 360.

    print("[INFO] Building KDTree")
    kdtree = cKDTree(np.c_[mlon, mlat], balanced_tree=False, compact_nodes=False) 

    # Align longitudes
    lon[lon < -180.0] += 360.
    lon[lon > 180.0] -= 360.

    ## Do KD Search:
    ll = np.ma.vstack([lon,lat]).T  # equiv. of np.c_ for masked arrays
    print("[INFO] Finding nearest neighbours for %d points." % np.alen(ll))
    dist, idx = kdtree.query(ll, distance_upper_bound=max_dist)
    ## possible to do... add in option for brute force distance based search

    # checking for locations I can't match up
    m = dist > max_dist
    if len(np.argwhere(m)) >= 1:
        print("[WARNING] %d" %len(np.argwhere(m)) + " request locations not matched up")
        for lp in np.argwhere(m):
            print("[WARNING] %8.4f" %lat[lp] + " %8.4f" %lon[lp])

    # either return compressed request and index lists
    if return_compressed:
        m = dist <= max_dist
        found_request = np.arange(len(lon))[m]
        if mindices.ndim == 1:
            found_indices = mindices[idx[m]]
        else:
            found_indices = mindices[idx[m],:]
        return found_request, found_indices

    # or index list with non-matched locations set as -1
    else:
        idx[m] = 0
        if mindices.ndim == 1:
            found_indices = mindices[idx]
            found_indices[m] = -1
        else:
            found_indices = mindices[idx,:]
            found_indices[m,:] = -1
        return found_indices


def matchupXY(varGrid, lons, lats, rngtol=10000., return_compressed=True):
    """Return matchup y,x indices for locations in 2D varGrid object"""

    latssp, lonssp, indices = sea_points_only_grid(varGrid)
    found_request, found_indices = nearest_indices(lonssp, latssp, indices, 
                                                   lons, lats, rngtol=rngtol,
                                                   return_compressed=True)

    return found_request, found_indices
