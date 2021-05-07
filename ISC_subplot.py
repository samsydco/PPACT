#!/usr/bin/env python3


# Plot differences in mean ISC between groups
# For subcortical ROIs
# Use: ISCdir+'ISCw_ISCb.h5'



import pandas as pd
import deepdish as dd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from settings import *


ISCbg = dd.io.load(ISCdir+'ISCw_ISCb.h5')
