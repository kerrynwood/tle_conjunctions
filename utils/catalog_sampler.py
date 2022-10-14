import numpy as np
from datetime import datetime, timedelta
import multiprocessing
import itertools
import time
import astropy.time
import numpy as np
import itertools
import sys
import random

from sgp4.io import twoline2rv
from sgp4.earth_gravity import wgs72
from sgp4.propagation import sgp4 as sgprop
from sgp4.conveniences import jday_datetime
from sgp4.ext import invjday

import multiprocessing

def outs( S ) : 
    sys.stderr.write( S )
    sys.stderr.flush()

def read_twoline_debug( fn ):
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
    return zip( L1, L2, tles )

# -----------------------------------------------------------------------------------------------------
# setup
EPOCH = astropy.time.Time('2022-10-01T00:00:00.000Z', format='isot')
EPJD  = EPOCH.jd
N     = 10000

outs('Loading catalog {}\n'.format(sys.argv[1]) )
tles = read_twoline_debug( sys.argv[1] )
# filter those near the epoch we want
outs('Filtering on epoch\n')
tles = list(filter( lambda X: abs(X[2].jdsatepoch - EPJD) < 30, tles ) )
outs('Sampling {} entries\n'.format(N))
tles = random.sample( tles, N )
outs('Resorting...\n')
tles = sorted( tles, key=lambda X: X[2].satnum )
outs('Writing...\n')

with open('./sample.tle','w') as F:
    for l1,l2,tle in tles:
        F.write(l1)
        F.write(l2)
