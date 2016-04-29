# -*- coding: utf-8 -*-
# -*- Python -*-

__all__ = []


"""
Progress bar

usage:

import time

nIter = 1000
nFlush = 51
t0 = time.time()
for i in xrange(nsim):
    time.sleep(0.001)
    updateProgress(i, nIter, 'test', nFlush, barLength=50)
print '\ntotal time : ', time.time() - t0

"""

import sys

def updateProgress(i, nIter, message='Progress', nFlush=50, barLength=50):
    # nFlush must be at max equal to nIter
    if nFlush > nIter:
        nFlush = nIter
    # update the progress bar only everyCheck
    everyCheck = round(float(nIter) / nFlush)
    if not ((i+1) % everyCheck) or (i+1) == nIter:
        progress = float(i+1) / nIter
        status = ""
        if progress >= 1:
            progress = 1
            status = "Done\r\n"
        block = int(round(barLength*progress))
        text =  "\r{0}: [{1}] {2}% {3}".format( message,"="*block + "-"*(barLength-block),
                                                    progress*100, status)
        sys.stdout.write(text)
        sys.stdout.flush()
