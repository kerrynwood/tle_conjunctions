import numpy as np
from datetime import datetime, timedelta
import multiprocessing
import itertools
import time

from sgp4.io import twoline2rv
from sgp4.earth_gravity import wgs72
from sgp4.propagation import sgp4 as sgprop
from sgp4.conveniences import jday_datetime

if __name__ == '__main__':
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
    with open( './catalog_20221001.twoline') as F: lines = F.readlines()
    L1 = list( filter( lambda X: X[0] == '1', lines ) )
    L2 = list( filter( lambda X: X[0] == '2', lines ) )

    output =''
    for a,b in zip( L1,L2 ):
        if a[18] == '2' and a[19] == '2': 
            output += a
            output += b
    with open('./catalog22.twoline','w') as F: F.write(output)
