# -*- coding: utf-8 -*-
# -*- Python -*-

__all__ = []


"""
Progress bar adapted from code found in
"http://stackoverflow.com/questions/3160699/python-progress-bar"

usage:

import time

print ""
print "progress : 0->1"
for i in xrange(100):
    time.sleep(0.1)
    updateProgress((i+1)/100.0)

"""

import sys

def updateProgress(progress, message, barLength=50):
    # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt\r\n"
    if progress >= 1:
        progress = 1
        status = "Done\r\n"
    block = int(round(barLength*progress))
    text =  "\r{0}: [{1}] {2}% {3}".format( message,"="*block + "-"*(barLength-block),
                                                progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()
