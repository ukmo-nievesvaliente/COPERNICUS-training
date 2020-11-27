# helper scripts for running and plotting isurf forecasts

import datetime as dt
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

import NearshoreAlgorithms as na
import Refrac_and_Surf as isurf

import wavetools.loaders.matchup as wvma
import wavetools.loaders.read_MOLevel1_wave as wvrd

def readBeachesRNLI(filein):
    """Read in beach metadata from file"""

    rdarray = np.genfromtxt(filein, skip_header=1, delimiter=',', usecols=[5,6,8,9,10,11,12])
    beaches = np.genfromtxt(filein, skip_header=1, delimiter=',', usecols=[4], dtype=str)
    sspaid  = np.genfromtxt(filein, skip_header=1, delimiter=',', usecols=[0], dtype=str)

    beachlats  = rdarray[:,0]
    beachlons  = rdarray[:,1]
    beachnorm  = rdarray[:,2]
    maxang     = rdarray[:,3]
    beachslope = rdarray[:,4]
    approslope = rdarray[:,5]
    beachtype  = np.array(rdarray[:,6], dtype=int)

    beachdata = {'sspaid':sspaid, 'name':beaches, 'lat':beachlats, 'lon':beachlons,
                 'normal':beachnorm, 'window':maxang, 'approachslope':approslope,
                 'beachslope':beachslope, 'type':beachtype}

    return beachdata


def readBeaches(filein):
    """Read in beach metadata from file"""

    rdarray = np.genfromtxt(filein, skip_header=1, delimiter=',', usecols=[5,6,8,9,10,11,12])
    beaches = np.genfromtxt(filein, skip_header=1, delimiter=',', usecols=[4], dtype=str)
    sspaid  = np.genfromtxt(filein, skip_header=1, delimiter=',', usecols=[0], dtype=str)

    beachlats  = rdarray[:,0]
    beachlons  = rdarray[:,1]
    beachnorm  = rdarray[:,2]
    maxang     = rdarray[:,3]
    beachslope = rdarray[:,4]
    approslope = rdarray[:,5]
    beachtype  = np.array(rdarray[:,6], dtype=int)

    return sspaid, beaches, beachlats, beachlons, beachnorm, maxang, approslope, beachslope, beachtype


def runMatchup(beachlats, beachlons, cycle, rngtol=10000, datadir='./'):

    hs = wvrd.readWaveL1('VHM0', cycle, leadtimes=None, domain='uk', datadir=datadir)    
    llmod, xymod = wvma.matchupXY(hs, beachlons, beachlats, rngtol=rngtol, return_compressed=True)

    return llmod, xymod


def readForecasts(cycle, leadtimes=None, xyindices=None, domain='uk', datadir='./'):
    """Create arrays containing the required wave time-series from file data"""

    print('[INFO] Assigning wave forecast data to arrays')
    wsGrid = wvrd.readWaveL1('WSPD', cycle, leadtimes=leadtimes, xyindices=xyindices, domain=domain, datadir=datadir)
    fctimes = wsGrid.fclead / 3600
    ws   = wsGrid.data
    wdir = wvrd.readWaveL1('WDIR', cycle, leadtimes=leadtimes, xyindices=xyindices, domain=domain, datadir=datadir).data
    hm0  = wvrd.readWaveL1('VHM0', cycle, leadtimes=leadtimes, xyindices=xyindices, domain=domain, datadir=datadir).data
    tp   = wvrd.readWaveL1('VTPK', cycle, leadtimes=leadtimes, xyindices=xyindices, domain=domain, datadir=datadir).data
    dirn = wvrd.readWaveL1('VMDR', cycle, leadtimes=leadtimes, xyindices=xyindices, domain=domain, datadir=datadir).data
    offdepth = wvrd.readWaveL1('deptho', cycle, leadtimes=None, xyindices=xyindices, domain=domain, datadir=datadir).data[0]

    hs_cmp  = np.zeros([np.shape(ws)[0],np.shape(ws)[1],3])
    tp_cmp  = np.zeros([np.shape(ws)[0],np.shape(ws)[1],3])
    dir_cmp = np.zeros([np.shape(ws)[0],np.shape(ws)[1],3])

    #offdepth = np.ones(np.shape(ws)[1])*50.0
    #offdepth = d.variables['dpt'][0,xymod]

    hs_cmp[:,:,0] = wvrd.readWaveL1('VHM0_WW', cycle, leadtimes=leadtimes, xyindices=xyindices, domain=domain, datadir=datadir).data
    hs_cmp[:,:,1] = wvrd.readWaveL1('VHM0_SW1', cycle, leadtimes=leadtimes, xyindices=xyindices, domain=domain, datadir=datadir).data
    hs_cmp[:,:,2] = wvrd.readWaveL1('VHM0_SW2', cycle, leadtimes=leadtimes, xyindices=xyindices, domain=domain, datadir=datadir).data
    tp_cmp[:,:,0] = wvrd.readWaveL1('VTPK_WW', cycle, leadtimes=leadtimes, xyindices=xyindices, domain=domain, datadir=datadir).data
    tp_cmp[:,:,1] = wvrd.readWaveL1('VTPK_SW1', cycle, leadtimes=leadtimes, xyindices=xyindices, domain=domain, datadir=datadir).data
    tp_cmp[:,:,2] = wvrd.readWaveL1('VTPK_SW2', cycle, leadtimes=leadtimes, xyindices=xyindices, domain=domain, datadir=datadir).data
    dir_cmp[:,:,0] = wvrd.readWaveL1('VMDR_WW', cycle, leadtimes=leadtimes, xyindices=xyindices, domain=domain, datadir=datadir).data
    dir_cmp[:,:,1] = wvrd.readWaveL1('VMDR_SW1', cycle, leadtimes=leadtimes, xyindices=xyindices, domain=domain, datadir=datadir).data
    dir_cmp[:,:,2] = wvrd.readWaveL1('VMDR_SW2', cycle, leadtimes=leadtimes, xyindices=xyindices, domain=domain, datadir=datadir).data

    # return zero values not missing data!
    hs_cmp = np.maximum(0.0, hs_cmp)
    tp_cmp = np.maximum(0.0, tp_cmp)
    dir_cmp = np.maximum(0.0, dir_cmp)

    offshoredata = {'depth':offdepth, 'wspd':ws, 'wdir':wdir, 'hm0':hm0, 'tp':tp, 'dirn':dirn, 'hscmp':hs_cmp, 'tpcmp':tp_cmp, 'dircmp':dir_cmp}

    return fctimes, offshoredata


def sitesXML(beachdata, outdir='.'):
    """Generate XML data for webpage viewer"""

    with open(outdir+'/surf_sites.xml','w') as outp:
        outp.write('<markers>\r\n')
        for isite in range(len(beachdata['name'])):
            outp.write('<marker lat="%6.3f' %beachdata['lat'][isite] + \
                       '" lng="%6.3f' %beachdata['lon'][isite] + \
                       '" name="' + beachdata['name'][isite].replace(' ','_').replace('/','-') + '"/>\r\n')
        outp.write('</markers>\r\n')
        outp.close()


def csvOutput(cycle, fctimes, beachdata, offshoredata, surfdata, fname='isurf_output.csv', outdir='.'):
    """Write diagnostic outputs to .csv file"""

    datestr = cycle.strftime('%Y%m%d00')

    with open(outdir+'/%s' %fname,'w') as outp:
        outp.write(datestr+'\r\n')
        for isite in range(len(beachdata['name'])):
            outp.write('\r\n')
            outp.write('%s' %beachdata['name'][isite] + '\r\n')
            outp.write('%d' %beachdata['type'][isite] + '\r\n')
            #outp.write('TI Hsmo Tpmo Dmo Hseq Tpeq DmEq Hsbr Dpbr\r\n')
            #outp.write('LT,Wspd,Wdir,Hsmo,Tpmo,Dmo,Tide,Hseq,Tpeq,DmEq,Hsbr,Dpbr,Hlbr,Hhbr,BT\r\n')
            outp.write('LT,Wspd,Wdir,Hsmo,Tpmo,Dmo,Hseq,Tpeq,DmEq,Hsbr,Dpbr,Hlbr,Hhbr,BT\r\n')

	        # write out to file
            for itime in range(len(fctimes)):

                # write out the data values to file
	            #outp.write ('%02d' %fctimes[lp] + ' %4.2f %4.1f %3d' %tuple([hm0[lp,isite], tp[lp,isite], dirn[lp,isite]]) + \
                #           ' %4.2f %4.1f %3d' %tuple([hsshwd[lp,isite], tpshwd[lp,isite], reldir[lp,isite]]) + ' %4.2f %4.2f' %tuple([hsbkinit[lp,isite], dpsat[lp,isite]]) + '\r\n')
	            outp.write('%02d' %fctimes[itime] + \
                           ',%4.1f' %offshoredata['wspd'][itime,isite] + \
                           #',%3d' %offshoredata['wdir'][itime,isite] + \
                           ',%4.2f' %offshoredata['hm0'][itime,isite] + \
                           ',%4.1f' %offshoredata['tp'][itime,isite] + \
                           ',%3d' %offshoredata['dirn'][itime,isite] + \
                           ',%4.2f' %surfdata['shorewardHs'][itime,isite] + \
                           ',%4.1f' %surfdata['shorewardT'][itime,isite] + \
                           ',%3d' %surfdata['relativeDirn'][itime,isite] + \
                           ',%4.2f' %surfdata['breakerHs'][itime,isite] + \
                           ',%4.2f' %surfdata['saturatedDepth'][itime,isite] + \
                           ',%4.2f' %surfdata['Hb1in3'][itime,isite] + \
                           ',%4.2f' %surfdata['Hb1in10'][itime,isite] + \
                           ',%1d' %surfdata['breakerType'][itime,isite] + '\r\n')
        outp.close()


def plotForecasts(cycle, fctimes, beachdata, offshoredata, surfdata, outdir='.'):
    """Plot the forecasts"""

    datestr = cycle.strftime('%Y%m%d00')

    for isite in range(len(beachdata['name'])):
        # plot the data
        print('Plotting data for '+beachdata['name'][isite])
        plt.figure( figsize=(8,9), facecolor='white' )

        #ax = plt.subplot(2,1,1)
        #ax.set_xticklabels( [] )
        #ax.set_xticks( np.arange(0,49,24) )
        #ms2kts = 3600.0 / 1853.0
        #u10 = ws[:,lps] * np.sin((wdir[:,lps]-180.0) * np.pi / 180.0)
        #v10 = ws[:,lps] * np.cos((wdir[:,lps]-180.0) * np.pi / 180.0)
        #print(u10,v10)
        #plt.barbs(fctimes, ws[:,lps], u10*ms2kts, v10*ms2kts, pivot='middle' )
        ##plt.barbs(date_list, ws[:,lps], u10*ms2kts, v10*ms2kts, pivot='middle' )
        #plt.xlim([0,48])
        #plt.ylim([0.0, np.max([20.0,np.max(ws[:,lps])+2.0])])
        #plt.ylabel( 'Wind Speed (m/s)' )

        ax = plt.subplot(1,1,1)
        ax.set_xticklabels( [] )
        ax.set_xticks( np.arange(0,49,24) )
        tmax = 21
        tmin = 3
        ymax = np.max([1.0,np.max(offshoredata['hm0'][:,isite])+0.2,np.max(surfdata['Hb1in10'][:,isite])+0.2])
        plt.fill_between(fctimes,surfdata['Hb1in3'][:,isite],surfdata['Hb1in10'][:,isite],color='#c3e4e5')
        hsu = np.sin((offshoredata['dirn'][:,isite]-180.0) * np.pi / 180.0)
        hsv = np.cos((offshoredata['dirn'][:,isite]-180.0) * np.pi / 180.0)
        plt.plot(fctimes,offshoredata['hm0'][:,isite],'k')
        qv = plt.quiver(fctimes, offshoredata['hm0'][:,isite], hsu, hsv, offshoredata['tp'][:,isite],
                        pivot='mid', units='inches', width=0.04, scale=6.0, scale_units='inches')
        plt.xlim([0,48])
        plt.ylim([0.0,ymax])
        plt.colorbar(qv,orientation='horizontal',extend='both',shrink=0.6)
        plt.clim(3.0,15.0)
        plt.ylabel('Wave/Surf height (m)')

        #ax = plt.subplot(3,1,3)
        #ax.xaxis.set_major_locator(dates.DayLocator())
        #ax.xaxis.set_major_formatter(dates.DateFormatter('%m-%dT%H:00'))
        #ax.set_xlim(date_list[0],date_list[-1])
        ##plt.plot(fctimes,fctide[:,lps],'k')
        #plt.plot(date_list,fctide[:,lps],'k')
        #ricoltxt = ['g','b','y','r','k']
        #ricol = []
        #for rc in ripind[:,lps]: ricol.append( ricoltxt[rc-1] )
        ##plt.scatter(fctimes,fctide[:,lps],color=ricol,s=35)
        #plt.scatter(date_list,fctide[:,lps],color=ricol,s=35)
        ##plt.xlim([0,48])
        ##plt.xticks(rotation='vertical')
        #plt.xlabel('Forecast time (UTC)')
        #plt.ylabel('Sea-surface height (m) / RI')

        plt.suptitle(beachdata['name'][isite]+'; shore normal: %003d' %beachdata['normal'][isite] + \
                     '; window: %0d' %beachdata['window'][isite] + '; type %d' %beachdata['type'][isite] + \
                     '; '+datestr[0:4]+'/'+datestr[4:6]+'/'+datestr[6:8])

        plt.savefig(outdir+'/'+beachdata['name'][isite].replace(' ','_').replace('/','-')+'.png')

        #plt.show()
        plt.close()

