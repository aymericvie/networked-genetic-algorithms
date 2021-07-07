# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 09:32:07 2021

@author: aymer
"""
import time
startTime = time.time()
b = 100000
e = 0
a = 0
while a < b:
    c = 0
    while c < 100:
        e += a + c
        c += 1
    a += 1
    print(a)
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
# took 13.24 in SPyder and 14.66 in Colab

import time
startTime = time.time()
b = 1000000
e = 0
a = 0
while a < b:
    c = 0
    while c < 100:
        e += a + c
        c += 1
    a += 1
    print(a)
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
# took 137.66 in SPyder/DELL and 155.91 in Colab
# took 94 sec in Spyder/ThinkPad
# took 68s on Enguerrand's PC i9 9900K
#Spyder/Dell 45% slower, Colab 64% slower
#Thinkpad: i5-10310U 1.7Ghz, 16GB RAM
#Dell: i7-8700, 3.2Ghz, 32GB RAM