import numpy as np
from datetime import datetime, timedelta
import multiprocessing
import itertools
import time
import astropy.time
import numpy as np
import itertools

from sgp4.io import twoline2rv
from sgp4.earth_gravity import wgs72
from sgp4.propagation import sgp4 as sgprop
from sgp4.conveniences import jday_datetime
from sgp4.ext import invjday

import multiprocessing

def read_twoline( fn ):
    '''
    reads in a file like:
    1 04523U 69082 GA 93237.23759955  .06028875 +36000-5 +45037-2 0  9999
    2 04523 070.0374 356.6819 0011002 283.3732 077.1114 16.15010855249410
    1 04524U 70071  A 70263.60375341  .00109633 +00000-0 +00000-0 0  9999
    2 04524 072.8586 118.4837 0110530 020.4754 340.0801 16.00284599001952
    1 04528U 70072  B 70256.72128638  .02030310 +00000-0 +00000-0 0  9991
    2 04528 051.5351 195.5582 0026658 320.0288 039.8725 16.26016213000198
    ....
    '''
    with open(fn) as F: lines = F.readlines()
    L1 = list( filter( lambda X: X[0] == '1', lines ) )
    L2 = list( filter( lambda X: X[0] == '2', lines ) )
    assert len(L1) == len(L2)
    tles = [ twoline2rv(A,B,wgs72) for A,B in zip(L1,L2) ]
    return tles

def propagate_tle( job ): 
    tleobj, jdates = job
    offsets = jdates - tleobj.jdsatepoch
    def propToJD( offset ): return np.hstack( sgprop( tleobj, offset ) )
    eph = np.vstack( [propToJD(O) for O in offsets ] )
    radius = np.linalg.norm( eph , axis=1 )
    return {'eph' : eph,
            'satnum'  : tleobj.satnum,
            'tleobj'  : tleobj,
            'dates'   : jdates,
            'step'    : np.max( np.diff( jdates ) ) * 86400,  # max step (in seconds)
            'apogee'  : np.max( radius ),
            'perigee' : np.min( radius ) }

def propagate_tle_wrapper( job ):
    try: return propagate_tle(job)
    except: return {'satnum' : job[0].satnum,
                    'error' : True }

def generate_catalog( tles, jdates, cores, debug=False ):
    jobs = [ (T,jdates) for T in tles ]
    N = len(tles)
    ephem = {}
    with multiprocessing.Pool( args.cores ) as pool:
        for i,result in enumerate(pool.imap_unordered( propagate_tle_wrapper, jobs )):
            pct = int(i/N * 100)
            if debug:
                print('[' + '*' * pct + '_' * (100-pct) + ']' + '{}'.format(pct) , end='\r')
            ephem[ result['satnum'] ] = result 
    if debug: print()
    return ephem

def conjunction_wrapper( job ):
    cat1, cat2, miss, debug = job 
    return conjunction_job( cat1, cat2, miss )

def conjunction_job( cat1, cat2, miss, buffer=200 ):
    if 'error' in cat1 or 'error' in cat2           : return []
    if cat1['satnum'] == cat2['satnum']             : return None
    if cat1['apogee'] + buffer < cat2['perigee']    : return None
    if cat2['apogee'] + buffer < cat1['perigee']    : return None
    # if ephemeris is time-aligned, we can numpy-vectorize
    x0    = cat1['eph'][:,0:3] - cat2['eph'][:,0:3]
    x0dot = cat1['eph'][:,3:] - cat2['eph'][:,3:]
    # equation 4 Healy
    tclos = np.einsum('ij,ij->i', x0, x0dot ) / np.einsum('ij,ij->i', x0dot, x0dot )
    # min dist (we'll filter for step size in a minute)
    mindist = x0 + (x0dot * tclos[:,np.newaxis])
    mindistM = np.linalg.norm( mindist, axis=1 )
    STEP     = np.max( (cat1['step'], cat2['step']) )
    # now find any that are within our step size (as first order approx, ignore those way outside step)
    # TODO: !!!! FIX THIS
    idx = np.where( (mindistM <= miss) & (np.abs(tclos) < STEP ) )[0]
    if len(idx) == 0: return None
    return [ { 'satnum1' : cat1['satnum'],
               'satnum2' : cat2['satnum'],
               'tclos'   : cat1['dates'][I] + (tclos[I]/86400.),
               'miss'    : mindistM[I] } for I in idx ]


# =====================================================================================================
if __name__ == '__main__':
    import sys
    import argparse
    import json
    DEFAULT_DAYS = 14

    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--cat1', help='First catalog of TLEs in two-line format', 
            required=True, type=str, dest='cat1')
    parser.add_argument('--cat2', help='Second catalog of TLEs in two-line format', 
            required=True, type=str, dest='cat2')
    parser.add_argument('--startdate', help='Start date for comparison, ex: 2022-01-01T00:00:00Z',
            required=True, type=str, dest='startdate')
    parser.add_argument('--enddate', help='End date for comparison (default +{} from start): ex : 2022-01-07T00:00:00Z '.format(DEFAULT_DAYS), 
            required=False, type=str, dest='enddate', default=None)
    parser.add_argument('--miss', help='Miss distance to output conjunctions (default: 20km)', 
            required=False, type=int, dest='miss', default=20)
    parser.add_argument('--step', help='Ephemeris step size (default 10 minutes )',
            required=False, type=int, dest='step', default=10)
    parser.add_argument('--processes', help='Use multiple cores',
            required=False, type=int, dest='cores', default=1)
    parser.add_argument('--out', help='Filename to output to',
            required=False, type=str, dest='outfile', default='./results.out')
    parser.add_argument('--debug', help='Debug output',
            action='store_true' )

    # Read arguments from command line
    args = parser.parse_args()
    sdate = datetime.strptime( args.startdate, '%Y-%m-%dT%H:%M:%S%z' )
    if args.enddate is not None:
        edate = datetime.strptime( args.enddate, '%Y-%m-%dT%H:%M:%S%z' )
    else:
        edate = sdate + timedelta( days = 14 )
    # some julian date holders
    sjd = sum( jday_datetime( sdate ) )
    ejd = sum( jday_datetime( edate ) )

    # output debugging info
    if args.debug: 
        print('Start date : {}  Julian : {}'.format( sdate, sjd) )
        print('End   date : {}  Julian : {}'.format( edate, ejd) )
        print('Cores      : {}'.format( args.cores ) )
        print('Outfile    : {}'.format( args.outfile ) )

    try: 
        with open( args.outfile,'w') as F: pass
    except Exception as e:
        print('Could not open the outfile, stopping run.  Error was : {}'.format(e))
        sys.exit()


    if args.debug : print('Loading cat1 ...', end='', flush=True) 
    tle1 = read_twoline( args.cat1 )
    if args.debug : print('... loaded {} TLEs'.format( len(tle1) ), flush=True )
    if args.debug : print('Loading cat2 ...', end='', flush=True) 
    tle2 = read_twoline( args.cat2 )
    if args.debug : print('... loaded {} TLEs'.format( len(tle2) ), flush=True )

    def filter_too_old( satrec ):
        if abs( satrec.jdsatepoch - sjd ) > 30 : 
            if args.debug: print('TLE epoch is {:10.2f} from start epoch, skipping '.format( 
                    satrec.satnum, satrec.jdsatepoch - sjd ) )
            return False
        return True
    tle1 = list( filter( filter_too_old, tle1 ) )
    tle2 = list( filter( filter_too_old, tle2 ) )

    # ===================================================================================================== DEBUG!!!!!!
    #tle1 = np.random.choice(tle1, 100, replace=False )
    #tle2 = np.random.choice(tle2, 100, replace=False )

    if args.debug:
        print('cat1: {} passed age check'.format(len(tle1)))
        print('cat2: {} passed age check'.format(len(tle2)))

    assert args.step > 0
    jdates = np.arange( sjd, ejd, args.step / 1440 )

    # ---------------------------------------------------------------------------------
    # def generate_catalog( tles, jdates, cores, debug=False ):
    if args.debug: print('Generating ephemeris for catalog 1...')
    catalog1 = generate_catalog( tle1, jdates, args.cores, args.debug )
    if args.debug: print('Generating ephemeris for catalog 2...')
    catalog2 = generate_catalog( tle2, jdates, args.cores, args.debug )

    # all vs all
    conj_jobs = itertools.product( catalog1.values(), catalog2.values(), [args.miss], [args.debug] )
    fhandle   = open( args.outfile, 'w' )
    total_jobs = len( catalog1.values() ) * len( catalog2.values() )
    found, complete   = 0,0
    with multiprocessing.Pool( args.cores ) as pool:
        for result in pool.imap_unordered( conjunction_wrapper, conj_jobs ):
            complete += 1
            print('Found: {}, {}/{}: {:5.2f}'.format(found,complete, total_jobs, 100 * complete/total_jobs), end='\r')
            if result is None: continue
            else: found += len( result )
            for record in result:
                fhandle.write( json.dumps( record ) + '\n')
    print()
    print()
