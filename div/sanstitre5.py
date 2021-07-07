# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 19:13:31 2021

@author: aymer
"""

from matplotlib import pyplot as plt
import numpy as np
# Create the data set
x = np.arange(0, 10, 0.05)
y = np.sin(x)
# Define the confidence interval
ci = 0.1 * np.std(y) / np.mean(y)
# Plot the sinus function
plt.plot(x, y)
# Plot the confidence interval
plt.fill_between(x, (y-ci), (y+ci), color='blue', alpha=0.1)
plt.show()